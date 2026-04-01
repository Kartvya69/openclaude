import { afterEach, beforeEach, expect, test } from 'bun:test'
import { createOpenAIShimClient } from './openaiShim.ts'

type FetchType = typeof globalThis.fetch

const originalEnv = {
  OPENAI_BASE_URL: process.env.OPENAI_BASE_URL,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
}

const originalFetch = globalThis.fetch

function makeSseResponse(lines: string[]): Response {
  const encoder = new TextEncoder()
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const line of lines) {
          controller.enqueue(encoder.encode(line))
        }
        controller.close()
      },
    }),
    {
      headers: {
        'Content-Type': 'text/event-stream',
      },
    },
  )
}

function makeJsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    headers: {
      'Content-Type': 'application/json',
    },
  })
}

function makeStreamChunks(chunks: unknown[]): string[] {
  return [
    ...chunks.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`),
    'data: [DONE]\n\n',
  ]
}

beforeEach(() => {
  process.env.OPENAI_BASE_URL = 'http://example.test/v1'
  process.env.OPENAI_API_KEY = 'test-key'
})

afterEach(() => {
  process.env.OPENAI_BASE_URL = originalEnv.OPENAI_BASE_URL
  process.env.OPENAI_API_KEY = originalEnv.OPENAI_API_KEY
  globalThis.fetch = originalFetch
})

test('preserves usage from final OpenAI stream chunk with empty choices', async () => {
  globalThis.fetch = (async (_input, init) => {
    const url = typeof _input === 'string' ? _input : _input.url
    expect(url).toBe('http://example.test/v1/chat/completions')

    const body = JSON.parse(String(init?.body))
    expect(body.stream).toBe(true)
    expect(body.stream_options).toEqual({ include_usage: true })

    const chunks = makeStreamChunks([
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: { role: 'assistant', content: 'hello world' },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'stop',
          },
        ],
      },
      {
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [],
        usage: {
          prompt_tokens: 123,
          completion_tokens: 45,
          total_tokens: 168,
        },
      },
    ])

    return makeSseResponse(chunks)
  }) as FetchType

  const client = createOpenAIShimClient({}) as {
    beta: {
      messages: {
        create: (
          params: Record<string, unknown>,
          options?: Record<string, unknown>,
        ) => Promise<unknown> & {
          withResponse: () => Promise<{ data: AsyncIterable<Record<string, unknown>> }>
        }
      }
    }
  }

  const result = await client.beta.messages
    .create({
      model: 'fake-model',
      system: 'test system',
      messages: [{ role: 'user', content: 'hello' }],
      max_tokens: 64,
      stream: true,
    })
    .withResponse()

  const events: Array<Record<string, unknown>> = []
  for await (const event of result.data) {
    events.push(event)
  }

  const usageEvent = events.find(
    event => event.type === 'message_delta' && typeof event.usage === 'object' && event.usage !== null,
  ) as { usage?: { input_tokens?: number; output_tokens?: number } } | undefined

  expect(usageEvent).toBeDefined()
  expect(usageEvent?.usage?.input_tokens).toBe(123)
  expect(usageEvent?.usage?.output_tokens).toBe(45)
})

test('reconciles misindexed tool call deltas from OpenAI-compatible streams', async () => {
  globalThis.fetch = (async (_input, init) => {
    const url = typeof _input === 'string' ? _input : _input.url
    expect(url).toBe('http://example.test/v1/chat/completions')

    const body = JSON.parse(String(init?.body))
    expect(body.stream).toBe(true)
    expect(body.tools?.[0]?.function?.name).toBe('Bash')

    const chunks = makeStreamChunks([
      {
        id: 'chatcmpl-tool-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: 'call_1',
                  type: 'function',
                  function: {
                    name: 'Bash',
                    arguments: '',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-tool-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 1,
                  function: {
                    arguments:
                      '{"command":"ls","description":"List project files"}',
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      {
        id: 'chatcmpl-tool-1',
        object: 'chat.completion.chunk',
        model: 'fake-model',
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: 'tool_calls',
          },
        ],
      },
    ])

    return makeSseResponse(chunks)
  }) as FetchType

  const client = createOpenAIShimClient({}) as {
    beta: {
      messages: {
        create: (
          params: Record<string, unknown>,
          options?: Record<string, unknown>,
        ) => Promise<unknown> & {
          withResponse: () => Promise<{ data: AsyncIterable<Record<string, unknown>> }>
        }
      }
    }
  }

  const result = await client.beta.messages
    .create({
      model: 'fake-model',
      system: 'test system',
      messages: [{ role: 'user', content: 'list files' }],
      max_tokens: 64,
      stream: true,
      tools: [
        {
          name: 'Bash',
          description: 'run shell commands',
          input_schema: {
            type: 'object',
            properties: {
              command: { type: 'string' },
              description: { type: 'string' },
            },
          },
        },
      ],
    })
    .withResponse()

  const events: Array<Record<string, unknown>> = []
  for await (const event of result.data) {
    events.push(event)
  }

  const toolStarts = events.filter(event => {
    const block = event.content_block as { type?: string } | undefined
    return event.type === 'content_block_start' && block?.type === 'tool_use'
  })
  expect(toolStarts).toHaveLength(1)

  const toolInputDelta = events.find(event => {
    const delta = event.delta as { type?: string } | undefined
    return event.type === 'content_block_delta' && delta?.type === 'input_json_delta'
  }) as { delta?: { partial_json?: string } } | undefined

  expect(toolInputDelta?.delta?.partial_json).toBe(
    '{"command":"ls","description":"List project files"}',
  )

  const stopEvent = events.find(
    event => event.type === 'message_delta',
  ) as { delta?: { stop_reason?: string } } | undefined
  expect(stopEvent?.delta?.stop_reason).toBe('tool_use')
})

test('forwards explicit GPT effort to chat completions as reasoning_effort', async () => {
  globalThis.fetch = (async (_input, init) => {
    const url = typeof _input === 'string' ? _input : _input.url
    expect(url).toBe('http://example.test/v1/chat/completions')

    const body = JSON.parse(String(init?.body))
    expect(body.model).toBe('gpt-5.4')
    expect(body.reasoning_effort).toBe('high')

    return makeJsonResponse({
      id: 'chatcmpl-reasoning-1',
      object: 'chat.completion',
      model: 'gpt-5.4',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'done',
          },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: 10,
        completion_tokens: 5,
      },
    })
  }) as FetchType

  const client = createOpenAIShimClient({}) as {
    beta: {
      messages: {
        create: (params: Record<string, unknown>) => Promise<unknown>
      }
    }
  }

  await client.beta.messages.create({
    model: 'gpt-5.4',
    system: 'test system',
    messages: [{ role: 'user', content: 'hello' }],
    max_tokens: 64,
    output_config: {
      effort: 'high',
    },
  })
})

test('uses GPT reasoning requested in the model descriptor for chat completions', async () => {
  globalThis.fetch = (async (_input, init) => {
    const body = JSON.parse(String(init?.body))
    expect(body.model).toBe('gpt-5.4')
    expect(body.reasoning_effort).toBe('medium')

    return makeJsonResponse({
      id: 'chatcmpl-reasoning-descriptor',
      object: 'chat.completion',
      model: 'gpt-5.4',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'done',
          },
          finish_reason: 'stop',
        },
      ],
    })
  }) as FetchType

  const client = createOpenAIShimClient({}) as {
    beta: {
      messages: {
        create: (params: Record<string, unknown>) => Promise<unknown>
      }
    }
  }

  await client.beta.messages.create({
    model: 'gpt-5.4?reasoning=medium',
    messages: [{ role: 'user', content: 'hello' }],
    max_tokens: 64,
  })
})

test('maps enabled GPT-5.4 thinking to low reasoning_effort by default', async () => {
  globalThis.fetch = (async (_input, init) => {
    const body = JSON.parse(String(init?.body))
    expect(body.model).toBe('gpt-5.4')
    expect(body.reasoning_effort).toBe('low')

    return makeJsonResponse({
      id: 'chatcmpl-reasoning-2',
      object: 'chat.completion',
      model: 'gpt-5.4',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'done',
          },
          finish_reason: 'stop',
        },
      ],
    })
  }) as FetchType

  const client = createOpenAIShimClient({}) as {
    beta: {
      messages: {
        create: (params: Record<string, unknown>) => Promise<unknown>
      }
    }
  }

  await client.beta.messages.create({
    model: 'gpt-5.4',
    messages: [{ role: 'user', content: 'hello' }],
    max_tokens: 64,
    thinking: {
      type: 'adaptive',
    },
  })
})

test('maps disabled thinking to the model-specific GPT reasoning off value', async () => {
  const observed: string[] = []

  globalThis.fetch = (async (_input, init) => {
    const body = JSON.parse(String(init?.body))
    observed.push(body.reasoning_effort)

    return makeJsonResponse({
      id: `chatcmpl-${observed.length}`,
      object: 'chat.completion',
      model: body.model,
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'done',
          },
          finish_reason: 'stop',
        },
      ],
    })
  }) as FetchType

  const client = createOpenAIShimClient({}) as {
    beta: {
      messages: {
        create: (params: Record<string, unknown>) => Promise<unknown>
      }
    }
  }

  await client.beta.messages.create({
    model: 'gpt-5.4',
    messages: [{ role: 'user', content: 'hello' }],
    max_tokens: 64,
    thinking: {
      type: 'disabled',
    },
  })

  await client.beta.messages.create({
    model: 'gpt-5',
    messages: [{ role: 'user', content: 'hello' }],
    max_tokens: 64,
    thinking: {
      type: 'disabled',
    },
  })

  expect(observed).toEqual(['none', 'minimal'])
})

test('maps max effort to xhigh for newer GPT chat models', async () => {
  globalThis.fetch = (async (_input, init) => {
    const body = JSON.parse(String(init?.body))
    expect(body.model).toBe('gpt-5.4')
    expect(body.reasoning_effort).toBe('xhigh')

    return makeJsonResponse({
      id: 'chatcmpl-reasoning-3',
      object: 'chat.completion',
      model: 'gpt-5.4',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: 'done',
          },
          finish_reason: 'stop',
        },
      ],
    })
  }) as FetchType

  const client = createOpenAIShimClient({}) as {
    beta: {
      messages: {
        create: (params: Record<string, unknown>) => Promise<unknown>
      }
    }
  }

  await client.beta.messages.create({
    model: 'gpt-5.4',
    messages: [{ role: 'user', content: 'hello' }],
    max_tokens: 64,
    output_config: {
      effort: 'max',
    },
  })
})
