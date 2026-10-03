import { useEffect, useState } from 'react'
import { useApi, type Status } from './api'
import { Icon } from './ui'
import { fmtInt, fmtUsd, fmtWhen } from './format'
import Overview from './screens/Overview'
import Process from './screens/Process'
import Insights from './screens/Insights'
import Review from './screens/Review'
import Evaluation from './screens/Evaluation'
import Cost from './screens/Cost'
import Gate from './screens/Gate'
import ThemeSwitch from './ThemeSwitch'

// Labels are the words the team uses out loud. "Evaluation" and "Run Cost"
// are what an engineer calls these screens; "How accurate is it" and "What it
// cost" are what a merchandiser asks.
const SCREENS = [
  { id: 'overview', label: 'This week', icon: 'overview' },
  { id: 'review', label: 'Your review', icon: 'review' },
  { id: 'insights', label: 'Dig deeper', icon: 'insights' },
  { id: 'process', label: 'Run the agent', icon: 'upload' },
  { id: 'evaluation', label: 'How accurate is it', icon: 'evaluation' },
  { id: 'cost', label: 'What it cost', icon: 'cost' },
] as const

type ScreenId = (typeof SCREENS)[number]['id']

function currentScreen(): ScreenId {
  const id = window.location.hash.replace(/^#\/?/, '')
  return (SCREENS.find((s) => s.id === id)?.id ?? 'overview') as ScreenId
}

export default function App() {
  const [screen, setScreen] = useState<ScreenId>(currentScreen)
  const status = useApi<Status>('/api/status')

  const { reload } = status
  useEffect(() => {
    const onHash = () => {
      setScreen(currentScreen())
      reload()
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [reload])

  // A run keeps going when you leave Upload & Process; keep the sidebar and
  // the gate in step with it so the dashboard opens as soon as it finishes.
  const jobRunning = status.data?.job?.state === 'running'
  useEffect(() => {
    if (!jobRunning) return
    const t = window.setInterval(reload, 2000)
    return () => window.clearInterval(t)
  }, [jobRunning, reload])

  useEffect(() => {
    const label = SCREENS.find((s) => s.id === screen)?.label
    document.title = `${label} · Dhaga & Co. Return Intelligence`
    document.querySelector('main')?.focus({ preventScroll: true })
    window.scrollTo(0, 0)
  }, [screen])

  const s = status.data
  const hasRun = !!s?.artifact
  const pending = s?.artifact ? s.review_pending : undefined

  const body = () => {
    if (screen === 'process') return <Process status={s} onChange={status.reload} />
    if (!s) return null
    if (!hasRun) return <Gate screen={SCREENS.find((x) => x.id === screen)!.label} />
    switch (screen) {
      case 'overview':
        return <Overview />
      case 'insights':
        return <Insights />
      case 'review':
        return <Review onChange={status.reload} />
      case 'evaluation':
        return <Evaluation />
      case 'cost':
        return <Cost />
    }
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Icon name="thread" size={20} />
          </span>
          <span>
            <span className="brand-name">Dhaga &amp; Co.</span>
            <br />
            <span className="brand-sub">Returns agent</span>
          </span>
        </div>
        <nav className="nav" aria-label="Screens">
          {SCREENS.map((item) => (
            <a key={item.id} href={`#/${item.id}`} aria-current={screen === item.id ? 'page' : undefined}>
              <Icon name={item.icon} />
              {item.label}
              {item.id === 'review' && hasRun && pending !== undefined && pending > 0 && (
                <span className="count" aria-label={`${pending} rows awaiting review`}>
                  {pending}
                </span>
              )}
            </a>
          ))}
        </nav>
        <ThemeSwitch />
        <div className="sidebar-foot">
          {hasRun && s?.run ? (
            <>
              <span>
                Last run: <span className="mono">{fmtInt(s.run.eligible_rows)}</span> comments for{' '}
                <span className="mono">{fmtUsd(s.run_total_usd, 2)}</span>
              </span>
              <span>Finished {fmtWhen(s.run.finished_at)}</span>
            </>
          ) : (
            <span>Nothing read yet. Upload a returns file to get started.</span>
          )}
          <span>Your uploads, run and corrections are private to this browser.</span>
        </div>
      </aside>
      <main className="main" tabIndex={-1} style={{ outline: 'none' }}>
        {status.error && screen !== 'process' ? (
          <p className="caption">Could not reach the API: {status.error.message}</p>
        ) : (
          body()
        )}
      </main>
    </div>
  )
}
