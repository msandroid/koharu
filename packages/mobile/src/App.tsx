import { useEffect, useMemo, useState } from 'react'

import * as api from './api'
import type { EngineEvent, Language, Model, Settings, Status } from './api'

const STAGES = ['detection', 'ocr', 'translation', 'inpainting'] as const
const STAGE_LABELS: Record<(typeof STAGES)[number], string> = {
  detection: 'Find text',
  ocr: 'Read text',
  translation: 'Translate',
  inpainting: 'Clean up',
}

type StageState = Extract<EngineEvent, { kind: 'stage' }>['state']
type Downloads = Record<string, { completed: number; total: number }>

export function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<'translate' | 'settings'>('translate')
  const [downloads, setDownloads] = useState<Downloads>({})
  const [stages, setStages] = useState<Partial<Record<string, StageState>>>({})

  useEffect(() => {
    const unlisten = api.onEngineEvent((event) => {
      if (event.kind === 'download') {
        setDownloads((current) => ({
          ...current,
          [event.name]: { completed: event.completed, total: event.total },
        }))
      } else if (event.kind === 'download_failed') {
        setError(`Download of ${event.name} failed: ${event.error}`)
      } else {
        setStages((current) => ({ ...current, [event.stage]: event.state }))
      }
    })
    api.status().then(setStatus, (reason) => setError(String(reason)))
    return () => {
      void unlisten.then((stop) => stop())
    }
  }, [])

  return (
    <div className='app'>
      <header>
        <h1>Koharu</h1>
        {status && (
          <button
            className='link'
            onClick={() => setView(view === 'settings' ? 'translate' : 'settings')}
          >
            {view === 'settings' ? 'Done' : 'Settings'}
          </button>
        )}
      </header>
      {error && (
        <p className='error' role='alert' onClick={() => setError(null)}>
          {error}
        </p>
      )}
      {!status ? (
        !error && <p className='muted'>Starting the translator…</p>
      ) : view === 'settings' ? (
        <SettingsPanel
          settings={status.settings}
          onSaved={(settings) => {
            setStatus({ ...status, settings })
            setView('translate')
          }}
          onError={setError}
        />
      ) : !status.models_ready ? (
        <ModelSetup
          downloads={downloads}
          onReady={() => setStatus({ ...status, models_ready: true })}
          onError={setError}
        />
      ) : (
        <TranslatePanel
          stages={stages}
          downloads={downloads}
          onStart={() => setStages({})}
          onError={setError}
        />
      )}
    </div>
  )
}

function ModelSetup({
  downloads,
  onReady,
  onError,
}: {
  downloads: Downloads
  onReady: () => void
  onError: (message: string) => void
}) {
  const [busy, setBusy] = useState(false)
  return (
    <section className='card'>
      <h2>Download the translator</h2>
      <p className='muted'>
        Koharu finds, reads, and erases text on your device. Its vision models take about 330 MB and
        are downloaded once.
      </p>
      <DownloadList downloads={downloads} />
      <button
        className='primary'
        disabled={busy}
        onClick={async () => {
          setBusy(true)
          try {
            await api.prepare()
            onReady()
          } catch (reason) {
            onError(String(reason))
          } finally {
            setBusy(false)
          }
        }}
      >
        {busy ? 'Downloading…' : 'Download models'}
      </button>
    </section>
  )
}

function TranslatePanel({
  stages,
  downloads,
  onStart,
  onError,
}: {
  stages: Partial<Record<string, StageState>>
  downloads: Downloads
  onStart: () => void
  onError: (message: string) => void
}) {
  const [page, setPage] = useState<File | null>(null)
  const [result, setResult] = useState<Blob | null>(null)
  const [busy, setBusy] = useState(false)
  const pageUrl = useObjectUrl(page)
  const resultUrl = useObjectUrl(result)

  return (
    <section className='card'>
      <label className='picker'>
        <input
          type='file'
          accept='image/*'
          onChange={(event) => {
            setPage(event.target.files?.[0] ?? null)
            setResult(null)
          }}
        />
        {page ? 'Choose another page' : 'Choose a manga page'}
      </label>
      {pageUrl && !resultUrl && <img className='page' src={pageUrl} alt='Selected page' />}
      {resultUrl && <img className='page' src={resultUrl} alt='Translated page' />}
      {page && (
        <button
          className='primary'
          disabled={busy}
          onClick={async () => {
            setBusy(true)
            onStart()
            try {
              setResult(await api.translate(page))
            } catch (reason) {
              onError(String(reason))
            } finally {
              setBusy(false)
            }
          }}
        >
          {busy ? 'Translating…' : result ? 'Translate again' : 'Translate'}
        </button>
      )}
      {busy && (
        <>
          <ol className='stages'>
            {STAGES.map((stage) => (
              <li key={stage} data-state={stages[stage] ?? 'waiting'}>
                {STAGE_LABELS[stage]}
              </li>
            ))}
          </ol>
          <DownloadList downloads={downloads} />
        </>
      )}
      {result && resultUrl && <ShareActions result={result} url={resultUrl} name={page?.name} />}
    </section>
  )
}

function ShareActions({ result, url, name }: { result: Blob; url: string; name?: string }) {
  const fileName = `${(name ?? 'page').replace(/\.[^.]+$/, '')}.translated.png`
  const file = useMemo(
    () => new File([result], fileName, { type: 'image/png' }),
    [result, fileName],
  )
  const canShare = typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })
  return (
    <div className='actions'>
      {canShare ? (
        <button onClick={() => void navigator.share({ files: [file] }).catch(() => {})}>
          Share
        </button>
      ) : (
        <a className='button' href={url} download={fileName}>
          Save
        </a>
      )}
    </div>
  )
}

function SettingsPanel({
  settings,
  onSaved,
  onError,
}: {
  settings: Settings
  onSaved: (settings: Settings) => void
  onError: (message: string) => void
}) {
  const [draft, setDraft] = useState(settings)
  const [apiKey, setApiKey] = useState('')
  const [models, setModels] = useState<Model[]>([])
  const [languages, setLanguages] = useState<Language[]>([])

  useEffect(() => {
    api.models().then(setModels, (reason) => onError(String(reason)))
    api.languages().then(setLanguages, (reason) => onError(String(reason)))
  }, [onError])

  const providers = [...new Set(models.map((model) => model.provider))]
  const providerModels = models.filter((model) => model.provider === draft.model.provider)
  const selected = providerModels.find((model) => model.model === draft.model.model)
  const local = draft.model.provider === 'local'

  return (
    <section className='card form'>
      <label>
        Translate into
        <select
          value={draft.target_language}
          onChange={(event) => setDraft({ ...draft, target_language: event.target.value })}
        >
          {languages.map((language) => (
            <option key={language.tag} value={language.tag}>
              {language.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Translator
        <select
          value={draft.model.provider}
          onChange={(event) => {
            const first = models.find((model) => model.provider === event.target.value)
            setDraft({
              ...draft,
              model: {
                provider: event.target.value,
                model: first?.model ?? null,
                quantization: first?.quantizations[0]?.id ?? null,
                vision: false,
                reasoning: false,
              },
            })
          }}
        >
          {providers.map((provider) => (
            <option key={provider} value={provider}>
              {provider === 'local' ? 'On this device' : provider}
            </option>
          ))}
        </select>
      </label>
      {providerModels.some((model) => model.model) && (
        <label>
          Model
          <select
            value={draft.model.model ?? ''}
            onChange={(event) => {
              const model = providerModels.find((entry) => entry.model === event.target.value)
              setDraft({
                ...draft,
                model: {
                  ...draft.model,
                  model: event.target.value,
                  quantization: model?.quantizations[0]?.id ?? null,
                },
              })
            }}
          >
            {providerModels.map((model) => (
              <option key={model.model ?? model.name} value={model.model ?? ''}>
                {model.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {local && selected && selected.quantizations.length > 1 && (
        <label>
          Size
          <select
            value={draft.model.quantization ?? ''}
            onChange={(event) =>
              setDraft({ ...draft, model: { ...draft.model, quantization: event.target.value } })
            }
          >
            {selected.quantizations.map((quantization) => (
              <option key={quantization.id} value={quantization.id}>
                {quantization.name}
                {quantization.downloaded ? ' (downloaded)' : ''}
              </option>
            ))}
          </select>
        </label>
      )}
      {!local && (
        <label>
          API key
          <input
            type='password'
            autoComplete='off'
            placeholder='Leave empty to keep the saved key'
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
          />
        </label>
      )}
      <button
        className='primary'
        onClick={async () => {
          try {
            await api.saveSettings(draft, apiKey || null)
            onSaved(draft)
          } catch (reason) {
            onError(String(reason))
          }
        }}
      >
        Save
      </button>
    </section>
  )
}

function DownloadList({ downloads }: { downloads: Downloads }) {
  const entries = Object.entries(downloads).filter(([, { completed, total }]) => completed < total)
  if (entries.length === 0) return null
  return (
    <ul className='downloads'>
      {entries.map(([name, { completed, total }]) => (
        <li key={name}>
          <span>{name}</span>
          <progress value={completed} max={total} />
        </li>
      ))}
    </ul>
  )
}

function useObjectUrl(blob: Blob | null) {
  const url = useMemo(() => (blob ? URL.createObjectURL(blob) : null), [blob])
  useEffect(() => () => void (url && URL.revokeObjectURL(url)), [url])
  return url
}
