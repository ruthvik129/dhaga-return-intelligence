import { useTheme, type ThemePref } from './theme'
import { Icon } from './ui'

const OPTIONS: { value: ThemePref; label: string; icon: string }[] = [
  { value: 'system', label: 'System', icon: 'monitor' },
  { value: 'light', label: 'Light', icon: 'sun' },
  { value: 'dark', label: 'Dark', icon: 'moon' },
]

/** System / Light / Dark, remembered in this browser. */
export default function ThemeSwitch() {
  const [pref, setPref] = useTheme()
  return (
    <div className="theme-switch" role="group" aria-label="Colour theme">
      {OPTIONS.map((o) => (
        <button
          key={o.value}
          type="button"
          aria-pressed={pref === o.value}
          title={`${o.label} theme`}
          onClick={() => setPref(o.value)}
        >
          <Icon name={o.icon} size={15} />
          <span className="theme-label">{o.label}</span>
        </button>
      ))}
    </div>
  )
}
