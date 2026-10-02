import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'

/** Mirrors `koharu_translator::ModelSelection`. */
export type ModelSelection = {
  provider: string
  model: string | null
  quantization: string | null
  vision: boolean
  reasoning: boolean
}

/** Mirrors `koharu_mobile_lib::Settings`. */
export type Settings = {
  target_language: string
  model: ModelSelection
}

export type Status = { models_ready: boolean; settings: Settings }

/** Mirrors `koharu_translator::Model`. */
export type Model = {
  provider: string
  model: string | null
  name: string
  quantizations: { id: string; name: string; downloaded: boolean }[]
  vision: boolean
  reasoning: boolean
}

export type Language = { tag: string; name: string }

/** Mirrors `koharu_mobile_lib::Event`. */
export type EngineEvent =
  | {
      kind: 'stage'
      stage: string
      model: string | null
      state: 'loading' | 'running' | 'finished' | 'skipped'
    }
  | { kind: 'download'; name: string; completed: number; total: number }
  | { kind: 'download_failed'; name: string; error: string }

export const status = () => invoke<Status>('status')
export const prepare = () => invoke<void>('prepare')
export const models = () => invoke<Model[]>('models')
export const languages = () => invoke<Language[]>('languages')
export const saveSettings = (settings: Settings, apiKey: string | null) =>
  invoke<void>('save_settings', { settings, apiKey })

/** Sends the encoded page as a raw body and receives the translated PNG. */
export async function translate(page: Blob): Promise<Blob> {
  const bytes = new Uint8Array(await page.arrayBuffer())
  const png = await invoke<ArrayBuffer>('translate', bytes)
  return new Blob([png], { type: 'image/png' })
}

export const onEngineEvent = (handler: (event: EngineEvent) => void): Promise<UnlistenFn> =>
  listen<EngineEvent>('koharu://event', (event) => handler(event.payload))
