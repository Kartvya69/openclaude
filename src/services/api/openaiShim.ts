/**
 * OpenAI-compatible API shim for Claude Code.
 *
 * Translates Anthropic SDK calls (anthropic.beta.messages.create) into
 * OpenAI-compatible chat completion requests and streams back events
 * in the Anthropic streaming format so the rest of the codebase is unaware.
 *
 * Supports: OpenAI, Azure OpenAI, Ollama, LM Studio, OpenRouter,
 * Together, Groq, Fireworks, DeepSeek, Mistral, and any OpenAI-compatible API.
 *
 * Environment variables:
 *   CLAUDE_CODE_USE_OPENAI=1          — enable this provider
 *   OPENAI_API_KEY=sk-...             — API key (optional for local models)
 *   OPENAI_BASE_URL=http://...        — base URL (default: https://api.openai.com/v1)
 *   OPENAI_MODEL=gpt-4o              — default model override
 *   CODEX_API_KEY / ~/.codex/auth.json — Codex auth for codexplan/codexspark
 */

import {
  codexStreamToAnthropic,
  collectCodexCompletedResponse,
  convertCodexResponseToAnthropicMessage,
  performCodexRequest,
  type AnthropicStreamEvent,
  type ShimCreateParams,
} from './codexShim.js'
import {
  resolveCodexApiCredentials,
  resolveProviderRequest,
} from './providerConfig.js'

// ---------------------------------------------------------------------------
// Types — minimal subset of Anthropic SDK types we need to produce
// ---------------------------------------------------------------------------

interface AnthropicUsage {
  input_tokens: number
  output_tokens: number
  cache_creation_input_tokens: number
  cache_read_input_tokens: number
}

// ---------------------------------------------------------------------------
// Message format conversion: Anthropic → OpenAI
// ---------------------------------------------------------------------------

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | Array<{ type: string; text?: string; image_url?: { url: string } }>
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
  }>
  tool_call_id?: string
  name?: string
}

interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
    strict?: boolean
  }
}

function convertSystemPrompt(
  system: unknown,
): string {
  if (!system) return ''
  if (typeof system === 'string') return system
  if (Array.isArray(system)) {
    return system
      .map((block: { type?: string; text?: string }) =>
        block.type === 'text' ? block.text ?? '' : '',
      )
      .join('\n\n')
  }
  return String(system)
}

function convertContentBlocks(
  content: unknown,
): string | Array<{ type: string; text?: string; image_url?: { url: string } }> {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return String(content ?? '')

  const parts: Array<{ type: string; text?: string; image_url?: { url: string } }> = []
  for (const block of content) {
    switch (block.type) {
      case 'text':
        parts.push({ type: 'text', text: block.text ?? '' })
        break
      case 'image': {
        const src = block.source
        if (src?.type === 'base64') {
          parts.push({
            type: 'image_url',
            image_url: {
              url: `data:${src.media_type};base64,${src.data}`,
            },
          })
        } else if (src?.type === 'url') {
          parts.push({ type: 'image_url', image_url: { url: src.url } })
        }
        break
      }
      case 'tool_use':
        // handled separately
        break
      case 'tool_result':
        // handled separately
        break
      case 'thinking':
        // Append thinking as text with a marker for models that support reasoning
        if (block.thinking) {
          parts.push({ type: 'text', text: `<thinking>${block.thinking}</thinking>` })
        }
        break
      default:
        if (block.text) {
          parts.push({ type: 'text', text: block.text })
        }
    }
  }

  if (parts.length === 0) return ''
  if (parts.length === 1 && parts[0].type === 'text') return parts[0].text ?? ''
  return parts
}

function convertMessages(
  messages: Array<{ role: string; message?: { role?: string; content?: unknown }; content?: unknown }>,
  system: unknown,
): OpenAIMessage[] {
  const result: OpenAIMessage[] = []

  // System message first
  const sysText = convertSystemPrompt(system)
  if (sysText) {
    result.push({ role: 'system', content: sysText })
  }

  for (const msg of messages) {
    // Claude Code wraps messages in { role, message: { role, content } }
    const inner = msg.message ?? msg
    const role = (inner as { role?: string }).role ?? msg.role
    const content = (inner as { content?: unknown }).content

    if (role === 'user') {
      // Check for tool_result blocks in user messages
      if (Array.isArray(content)) {
        const toolResults = content.filter((b: { type?: string }) => b.type === 'tool_result')
        const otherContent = content.filter((b: { type?: string }) => b.type !== 'tool_result')

        // Emit tool results as tool messages
        for (const tr of toolResults) {
          const trContent = Array.isArray(tr.content)
            ? tr.content.map((c: { text?: string }) => c.text ?? '').join('\n')
            : typeof tr.content === 'string'
              ? tr.content
              : JSON.stringify(tr.content ?? '')
          result.push({
            role: 'tool',
            tool_call_id: tr.tool_use_id ?? 'unknown',
            content: tr.is_error ? `Error: ${trContent}` : trContent,
          })
        }

        // Emit remaining user content
        if (otherContent.length > 0) {
          result.push({
            role: 'user',
            content: convertContentBlocks(otherContent),
          })
        }
      } else {
        result.push({
          role: 'user',
          content: convertContentBlocks(content),
        })
      }
    } else if (role === 'assistant') {
      // Check for tool_use blocks
      if (Array.isArray(content)) {
        const toolUses = content.filter((b: { type?: string }) => b.type === 'tool_use')
        const textContent = content.filter(
          (b: { type?: string }) => b.type !== 'tool_use' && b.type !== 'thinking',
        )

        const assistantMsg: OpenAIMessage = {
          role: 'assistant',
          content: convertContentBlocks(textContent) as string,
        }

        if (toolUses.length > 0) {
          assistantMsg.tool_calls = toolUses.map(
            (tu: { id?: string; name?: string; input?: unknown }) => ({
              id: tu.id ?? `call_${Math.random().toString(36).slice(2)}`,
              type: 'function' as const,
              function: {
                name: tu.name ?? 'unknown',
                arguments:
                  typeof tu.input === 'string'
                    ? tu.input
                    : JSON.stringify(tu.input ?? {}),
              },
            }),
          )
        }

        result.push(assistantMsg)
      } else {
        result.push({
          role: 'assistant',
          content: convertContentBlocks(content) as string,
        })
      }
    }
  }

  return result
}

function convertTools(
  tools: Array<{ name: string; description?: string; input_schema?: Record<string, unknown> }>,
): OpenAITool[] {
  return tools
    .filter(t => t.name !== 'ToolSearchTool') // Not relevant for OpenAI
    .map(t => ({
      type: 'function' as const,
      function: {
        name: t.name,
        description: t.description ?? '',
        parameters: t.input_schema ?? { type: 'object', properties: {} },
      },
    }))
}

type OpenAIReasoningEffort =
  | 'none'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh'

type OpenAIReasoningProfile = {
  allowed: OpenAIReasoningEffort[]
  disabledEffort?: OpenAIReasoningEffort
  defaultEffort?: OpenAIReasoningEffort
}

function getOpenAIReasoningProfile(model: string): OpenAIReasoningProfile | undefined {
  const normalized = model.trim().toLowerCase()

  if (normalized.includes('gpt-oss')) {
    return {
      allowed: ['low', 'medium', 'high'],
      defaultEffort: 'medium',
    }
  }

  if (!normalized.includes('gpt-5')) {
    return undefined
  }

  if (normalized.includes('gpt-5-pro')) {
    return {
      allowed: ['high'],
      defaultEffort: 'high',
    }
  }

  if (normalized.includes('-pro')) {
    return {
      allowed: ['medium', 'high', 'xhigh'],
      defaultEffort: 'medium',
    }
  }

  if (normalized.includes('-codex')) {
    return {
      allowed: ['low', 'medium', 'high', 'xhigh'],
      defaultEffort: 'medium',
    }
  }

  if (normalized.startsWith('gpt-5.1')) {
    return {
      allowed: ['none', 'low', 'medium', 'high'],
      disabledEffort: 'none',
      defaultEffort: 'none',
    }
  }

  if (normalized.startsWith('gpt-5.')) {
    return {
      allowed: ['none', 'low', 'medium', 'high', 'xhigh'],
      disabledEffort: 'none',
      defaultEffort: 'none',
    }
  }

  return {
    allowed: ['minimal', 'low', 'medium', 'high'],
    disabledEffort: 'minimal',
    defaultEffort: 'medium',
  }
}

function clampOpenAIReasoningEffort(
  profile: OpenAIReasoningProfile,
  effort: string,
): OpenAIReasoningEffort | undefined {
  const normalized = effort.trim().toLowerCase()
  if (!normalized) return undefined

  const direct = normalized as OpenAIReasoningEffort
  if (profile.allowed.includes(direct)) {
    return direct
  }

  if (normalized === 'max') {
    return profile.allowed.includes('xhigh') ? 'xhigh' : 'high'
  }

  const rankedEfforts: OpenAIReasoningEffort[] = [
    'none',
    'minimal',
    'low',
    'medium',
    'high',
    'xhigh',
  ]
  const requestedRank = rankedEfforts.indexOf(direct)
  if (requestedRank === -1) return undefined

  const allowedRanked = profile.allowed
    .filter(candidate => rankedEfforts.includes(candidate))
    .sort(
      (left, right) => rankedEfforts.indexOf(left) - rankedEfforts.indexOf(right),
    )
  const highestAllowed = allowedRanked[allowedRanked.length - 1]

  return (
    allowedRanked.find(candidate => rankedEfforts.indexOf(candidate) >= requestedRank) ??
    highestAllowed
  )
}

function readOutputConfigEffort(params: ShimCreateParams): string | undefined {
  const outputConfig = params.output_config
  if (!outputConfig || typeof outputConfig !== 'object' || Array.isArray(outputConfig)) {
    return undefined
  }
  const effort = (outputConfig as { effort?: unknown }).effort
  return typeof effort === 'string' && effort.trim() ? effort : undefined
}

function isThinkingDisabled(thinking: unknown): boolean {
  if (thinking === false) return true
  if (!thinking || typeof thinking !== 'object' || Array.isArray(thinking)) {
    return false
  }
  return (thinking as { type?: unknown }).type === 'disabled'
}

function isThinkingEnabled(thinking: unknown): boolean {
  if (!thinking || typeof thinking !== 'object' || Array.isArray(thinking)) {
    return false
  }
  const type = (thinking as { type?: unknown }).type
  return type === 'adaptive' || type === 'enabled'
}

function resolveOpenAIChatReasoningEffort(
  request: ReturnType<typeof resolveProviderRequest>,
  params: ShimCreateParams,
): OpenAIReasoningEffort | undefined {
  const profile = getOpenAIReasoningProfile(request.resolvedModel)
  if (!profile) return undefined

  if (isThinkingDisabled(params.thinking)) {
    return profile.disabledEffort
  }

  const explicitEffort =
    readOutputConfigEffort(params) ?? request.reasoning?.effort
  if (explicitEffort) {
    return clampOpenAIReasoningEffort(profile, explicitEffort)
  }

  if (isThinkingEnabled(params.thinking) && profile.defaultEffort === 'none') {
    return clampOpenAIReasoningEffort(profile, 'low')
  }

  return undefined
}

// ---------------------------------------------------------------------------
// Streaming: OpenAI SSE → Anthropic stream events
// ---------------------------------------------------------------------------

interface OpenAIStreamChunk {
  id: string
  object: string
  model: string
  choices: Array<{
    index: number
    delta: {
      role?: string
      content?: string | null
      tool_calls?: Array<{
        index: number
        id?: string
        type?: string
        function?: { name?: string; arguments?: string }
      }>
    }
    finish_reason: string | null
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  }
}

type ToolSchemaHints = Map<string, Set<string>>

type ActiveToolCall = {
  blockIndex: number
  providerIndex?: number
  id?: string
  name?: string
  bestArguments: string
  startedOrder: number
}

function makeMessageId(): string {
  return `msg_${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`
}

function convertChunkUsage(
  usage: OpenAIStreamChunk['usage'] | undefined,
): Partial<AnthropicUsage> | undefined {
  if (!usage) return undefined

  return {
    input_tokens: usage.prompt_tokens ?? 0,
    output_tokens: usage.completion_tokens ?? 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  }
}
function tryParseJSON(input: string): unknown | null {
  if (!input) return null
  try {
    return JSON.parse(input)
  } catch {
    return null
  }
}

function buildToolSchemaHints(
  tools?: Array<{ name?: string; input_schema?: Record<string, unknown> }>,
): ToolSchemaHints {
  const hints: ToolSchemaHints = new Map()
  for (const tool of tools ?? []) {
    if (!tool.name) continue
    const properties = tool.input_schema?.properties
    if (!properties || typeof properties !== 'object') continue
    hints.set(tool.name, new Set(Object.keys(properties)))
  }
  return hints
}

function getJSONInputKeys(input: string): string[] | null {
  const parsed = tryParseJSON(input)
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
    return null
  }
  return Object.keys(parsed as Record<string, unknown>)
}

function mergeToolArguments(current: string, incoming: string): string {
  if (!incoming) return current
  if (!current) return incoming
  if (incoming === current) return current

  const incomingParsed = tryParseJSON(incoming)
  const currentParsed = tryParseJSON(current)
  if (incomingParsed !== null) {
    if (currentParsed === null || incoming.length >= current.length) {
      return incoming
    }
    return current
  }

  if (incoming.startsWith(current)) return incoming
  if (current.startsWith(incoming)) return current

  const appended = current + incoming
  return appended.length >= current.length ? appended : current
}

function scoreToolCallCandidate(
  candidate: ActiveToolCall,
  chunk: {
    index?: number
    id?: string
    function?: { name?: string; arguments?: string }
  },
  onlyOneToolCallInChunk: boolean,
  lastStartedToolCall: ActiveToolCall | undefined,
  toolSchemaHints: ToolSchemaHints,
): number {
  let score = 0
  const incomingArgs = chunk.function?.arguments ?? ''

  if (chunk.id && candidate.id === chunk.id) score += 10_000
  if (
    typeof chunk.index === 'number' &&
    typeof candidate.providerIndex === 'number' &&
    candidate.providerIndex === chunk.index
  ) {
    score += 100
  }
  if (chunk.function?.name && candidate.name === chunk.function.name) {
    score += 25
  }
  if (candidate === lastStartedToolCall && onlyOneToolCallInChunk) {
    score += 35
  }

  if (incomingArgs) {
    if (candidate.bestArguments === incomingArgs) {
      score += 1_000
    } else if (
      candidate.bestArguments.length > 0 &&
      incomingArgs.startsWith(candidate.bestArguments)
    ) {
      score += 250
    } else if (
      incomingArgs.length > 0 &&
      candidate.bestArguments.startsWith(incomingArgs)
    ) {
      score += 150
    } else if (candidate.bestArguments.length === 0) {
      score += 40
    }

    const inputKeys = getJSONInputKeys(incomingArgs)
    if (inputKeys && candidate.name) {
      const knownKeys = toolSchemaHints.get(candidate.name)
      if (knownKeys) {
        let overlap = 0
        for (const key of inputKeys) {
          if (knownKeys.has(key)) overlap++
        }
        if (overlap > 0) {
          score += overlap * 200
          score -= (inputKeys.length - overlap) * 50
        }
      }
    }
  }

  return score
}

function resolveToolCallCandidate(
  activeToolCalls: ActiveToolCall[],
  chunk: {
    index?: number
    id?: string
    function?: { name?: string; arguments?: string }
  },
  onlyOneToolCallInChunk: boolean,
  lastStartedToolCall: ActiveToolCall | undefined,
  toolSchemaHints: ToolSchemaHints,
): ActiveToolCall | undefined {
  let best: { call: ActiveToolCall; score: number } | undefined

  for (const candidate of activeToolCalls) {
    const score = scoreToolCallCandidate(
      candidate,
      chunk,
      onlyOneToolCallInChunk,
      lastStartedToolCall,
      toolSchemaHints,
    )
    if (
      !best ||
      score > best.score ||
      (score === best.score &&
        candidate.startedOrder > best.call.startedOrder)
    ) {
      best = { call: candidate, score }
    }
  }

  return best?.score && best.score > 0 ? best.call : undefined
}
/**
 * Async generator that transforms an OpenAI SSE stream into
 * Anthropic-format BetaRawMessageStreamEvent objects.
 */
async function* openaiStreamToAnthropic(
  response: Response,
  model: string,
  tools?: Array<{ name?: string; input_schema?: Record<string, unknown> }>,
): AsyncGenerator<AnthropicStreamEvent> {
  const messageId = makeMessageId()
  let contentBlockIndex = 0
  const activeToolCalls: ActiveToolCall[] = []
  const activeToolCallsById = new Map<string, ActiveToolCall>()
  const activeToolCallsByProviderIndex = new Map<number, ActiveToolCall>()
  const toolSchemaHints = buildToolSchemaHints(tools)
  let nextToolCallOrder = 0
  let lastStartedToolCall: ActiveToolCall | undefined
  let hasEmittedContentStart = false
  let lastStopReason: 'tool_use' | 'max_tokens' | 'end_turn' | null = null
  let hasEmittedFinalUsage = false

  // Emit message_start
  yield {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    },
  }

  const reader = response.body?.getReader()
  if (!reader) return

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed === 'data: [DONE]') continue
      if (!trimmed.startsWith('data: ')) continue

      let chunk: OpenAIStreamChunk
      try {
        chunk = JSON.parse(trimmed.slice(6))
      } catch {
        continue
      }

      const chunkUsage = convertChunkUsage(chunk.usage)

      for (const choice of chunk.choices ?? []) {
        const delta = choice.delta

        // Text content
        if (delta.content) {
          if (!hasEmittedContentStart) {
            yield {
              type: 'content_block_start',
              index: contentBlockIndex,
              content_block: { type: 'text', text: '' },
            }
            hasEmittedContentStart = true
          }
          yield {
            type: 'content_block_delta',
            index: contentBlockIndex,
            delta: { type: 'text_delta', text: delta.content },
          }
        }

        // Tool calls
        if (delta.tool_calls) {
          const onlyOneToolCallInChunk = delta.tool_calls.length === 1
          for (const tc of delta.tool_calls) {
            const incomingArgs = tc.function?.arguments ?? ''
            const hasIdentity = Boolean(tc.id && tc.function?.name)
            let activeToolCall: ActiveToolCall | undefined

            if (hasIdentity) {
              activeToolCall =
                (tc.id ? activeToolCallsById.get(tc.id) : undefined) ??
                (typeof tc.index === 'number'
                  ? activeToolCallsByProviderIndex.get(tc.index)
                  : undefined)

              if (!activeToolCall) {
                // New tool call starting
                if (hasEmittedContentStart) {
                  yield {
                    type: 'content_block_stop',
                    index: contentBlockIndex,
                  }
                  hasEmittedContentStart = false
                }

                activeToolCall = {
                  blockIndex: contentBlockIndex,
                  providerIndex: tc.index,
                  id: tc.id,
                  name: tc.function?.name,
                  bestArguments: '',
                  startedOrder: nextToolCallOrder++,
                }
                activeToolCalls.push(activeToolCall)
                lastStartedToolCall = activeToolCall

                yield {
                  type: 'content_block_start',
                  index: activeToolCall.blockIndex,
                  content_block: {
                    type: 'tool_use',
                    id: tc.id!,
                    name: tc.function!.name!,
                    input: {},
                  },
                }
                contentBlockIndex++
              }

              if (tc.id) {
                activeToolCall.id = tc.id
                activeToolCallsById.set(tc.id, activeToolCall)
              }
              if (tc.function?.name) {
                activeToolCall.name = tc.function.name
              }
              if (typeof tc.index === 'number') {
                activeToolCall.providerIndex = tc.index
                activeToolCallsByProviderIndex.set(tc.index, activeToolCall)
              }
            } else if (incomingArgs) {
              activeToolCall = resolveToolCallCandidate(
                activeToolCalls,
                tc,
                onlyOneToolCallInChunk,
                lastStartedToolCall,
                toolSchemaHints,
              )
            }

            if (activeToolCall && incomingArgs) {
              activeToolCall.bestArguments = mergeToolArguments(
                activeToolCall.bestArguments,
                incomingArgs,
              )
            }
          }
        }

        // Finish
        if (choice.finish_reason) {
          // Close any open content blocks
          if (hasEmittedContentStart) {
            yield {
              type: 'content_block_stop',
              index: contentBlockIndex,
            }
            hasEmittedContentStart = false
          }

          // Emit the reconciled tool arguments once, then close each tool block.
          for (const tc of activeToolCalls.sort(
            (a, b) => a.blockIndex - b.blockIndex,
          )) {
            if (tc.bestArguments) {
              yield {
                type: 'content_block_delta',
                index: tc.blockIndex,
                delta: {
                  type: 'input_json_delta',
                  partial_json: tc.bestArguments,
                },
              }
            }
            yield { type: 'content_block_stop', index: tc.blockIndex }
          }

          const stopReason =
            choice.finish_reason === 'tool_calls'
              ? 'tool_use'
              : choice.finish_reason === 'length'
                ? 'max_tokens'
                : 'end_turn'
          lastStopReason = stopReason

          yield {
            type: 'message_delta',
            delta: { stop_reason: stopReason, stop_sequence: null },
            ...(chunkUsage ? { usage: chunkUsage } : {}),
          }
          if (chunkUsage) {
            hasEmittedFinalUsage = true
          }

          activeToolCalls.length = 0
          activeToolCallsById.clear()
          activeToolCallsByProviderIndex.clear()
          lastStartedToolCall = undefined
        }
      }

      if (
        !hasEmittedFinalUsage &&
        chunkUsage &&
        (chunk.choices?.length ?? 0) === 0
      ) {
        yield {
          type: 'message_delta',
          delta: { stop_reason: lastStopReason, stop_sequence: null },
          usage: chunkUsage,
        }
        hasEmittedFinalUsage = true
      }
    }
  }

  yield { type: 'message_stop' }
}

// ---------------------------------------------------------------------------
// The shim client — duck-types as Anthropic SDK
// ---------------------------------------------------------------------------

class OpenAIShimStream {
  private generator: AsyncGenerator<AnthropicStreamEvent>
  // The controller property is checked by claude.ts to distinguish streams from error messages
  controller = new AbortController()

  constructor(generator: AsyncGenerator<AnthropicStreamEvent>) {
    this.generator = generator
  }

  async *[Symbol.asyncIterator]() {
    yield* this.generator
  }
}

class OpenAIShimMessages {
  private defaultHeaders: Record<string, string>

  constructor(defaultHeaders: Record<string, string>) {
    this.defaultHeaders = defaultHeaders
  }

  create(
    params: ShimCreateParams,
    options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ) {
    const self = this

    const promise = (async () => {
      const request = resolveProviderRequest({ model: params.model })
      const response = await self._doRequest(request, params, options)

      if (params.stream) {
        return new OpenAIShimStream(
          request.transport === 'codex_responses'
            ? codexStreamToAnthropic(response, request.resolvedModel)
            : openaiStreamToAnthropic(
                response,
                request.resolvedModel,
                params.tools as
                  | Array<{ name?: string; input_schema?: Record<string, unknown> }>
                  | undefined,
              ),
        )
      }

      if (request.transport === 'codex_responses') {
        const data = await collectCodexCompletedResponse(response)
        return convertCodexResponseToAnthropicMessage(
          data,
          request.resolvedModel,
        )
      }

      const data = await response.json()
      return self._convertNonStreamingResponse(data, request.resolvedModel)
    })()

    ;(promise as unknown as Record<string, unknown>).withResponse =
      async () => {
        const data = await promise
        return {
          data,
          response: new Response(),
          request_id: makeMessageId(),
        }
      }

    return promise
  }

  private async _doRequest(
    request: ReturnType<typeof resolveProviderRequest>,
    params: ShimCreateParams,
    options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ): Promise<Response> {
    if (request.transport === 'codex_responses') {
      const credentials = resolveCodexApiCredentials()
      if (!credentials.apiKey) {
        const authHint = credentials.authPath
          ? ` or place a Codex auth.json at ${credentials.authPath}`
          : ''
        throw new Error(
          `Codex auth is required for ${request.requestedModel}. Set CODEX_API_KEY${authHint}.`,
        )
      }
      if (!credentials.accountId) {
        throw new Error(
          'Codex auth is missing chatgpt_account_id. Re-login with the Codex CLI or set CHATGPT_ACCOUNT_ID/CODEX_ACCOUNT_ID.',
        )
      }

      return performCodexRequest({
        request,
        credentials,
        params,
        defaultHeaders: {
          ...this.defaultHeaders,
          ...(options?.headers ?? {}),
        },
        signal: options?.signal,
      })
    }

    return this._doOpenAIRequest(request, params, options)
  }

  private async _doOpenAIRequest(
    request: ReturnType<typeof resolveProviderRequest>,
    params: ShimCreateParams,
    options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ): Promise<Response> {
    const openaiMessages = convertMessages(
      params.messages as Array<{
        role: string
        message?: { role?: string; content?: unknown }
        content?: unknown
      }>,
      params.system,
    )

    const body: Record<string, unknown> = {
      model: request.resolvedModel,
      messages: openaiMessages,
      max_tokens: params.max_tokens,
      stream: params.stream ?? false,
    }

    if (params.stream) {
      body.stream_options = { include_usage: true }
    }

    if (params.temperature !== undefined) body.temperature = params.temperature
    if (params.top_p !== undefined) body.top_p = params.top_p

    const reasoningEffort = resolveOpenAIChatReasoningEffort(request, params)
    if (reasoningEffort) {
      body.reasoning_effort = reasoningEffort
    }

    if (params.tools && params.tools.length > 0) {
      const converted = convertTools(
        params.tools as Array<{
          name: string
          description?: string
          input_schema?: Record<string, unknown>
        }>,
      )
      if (converted.length > 0) {
        body.tools = converted
        if (params.tool_choice) {
          const tc = params.tool_choice as { type?: string; name?: string }
          if (tc.type === 'auto') {
            body.tool_choice = 'auto'
          } else if (tc.type === 'tool' && tc.name) {
            body.tool_choice = {
              type: 'function',
              function: { name: tc.name },
            }
          } else if (tc.type === 'any') {
            body.tool_choice = 'required'
          }
        }
      }
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.defaultHeaders,
      ...(options?.headers ?? {}),
    }

    const apiKey = process.env.OPENAI_API_KEY ?? ''
    if (apiKey) {
      headers.Authorization = `Bearer ${apiKey}`
    }

    const response = await fetch(`${request.baseUrl}/chat/completions`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: options?.signal,
    })

    if (!response.ok) {
      const errorBody = await response.text().catch(() => 'unknown error')
      throw new Error(`OpenAI API error ${response.status}: ${errorBody}`)
    }

    return response
  }

  private _convertNonStreamingResponse(
    data: {
      id?: string
      model?: string
      choices?: Array<{
        message?: {
          role?: string
          content?: string | null
          tool_calls?: Array<{
            id: string
            function: { name: string; arguments: string }
          }>
        }
        finish_reason?: string
      }>
      usage?: {
        prompt_tokens?: number
        completion_tokens?: number
      }
    },
    model: string,
  ) {
    const choice = data.choices?.[0]
    const content: Array<Record<string, unknown>> = []

    if (choice?.message?.content) {
      content.push({ type: 'text', text: choice.message.content })
    }

    if (choice?.message?.tool_calls) {
      for (const tc of choice.message.tool_calls) {
        let input: unknown
        try {
          input = JSON.parse(tc.function.arguments)
        } catch {
          input = { raw: tc.function.arguments }
        }
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.function.name,
          input,
        })
      }
    }

    const stopReason =
      choice?.finish_reason === 'tool_calls'
        ? 'tool_use'
        : choice?.finish_reason === 'length'
          ? 'max_tokens'
          : 'end_turn'

    return {
      id: data.id ?? makeMessageId(),
      type: 'message',
      role: 'assistant',
      content,
      model: data.model ?? model,
      stop_reason: stopReason,
      stop_sequence: null,
      usage: {
        input_tokens: data.usage?.prompt_tokens ?? 0,
        output_tokens: data.usage?.completion_tokens ?? 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    }
  }
}

class OpenAIShimBeta {
  messages: OpenAIShimMessages

  constructor(defaultHeaders: Record<string, string>) {
    this.messages = new OpenAIShimMessages(defaultHeaders)
  }
}

export function createOpenAIShimClient(options: {
  defaultHeaders?: Record<string, string>
  maxRetries?: number
  timeout?: number
}): unknown {
  const beta = new OpenAIShimBeta({
    ...(options.defaultHeaders ?? {}),
  })

  return {
    beta,
    messages: beta.messages,
  }
}
