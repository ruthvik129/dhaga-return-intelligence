// Theme preference: "system" follows the OS, "light"/"dark" override it.
// The resolved theme is written to <html data-theme>, which index.css keys
// the dark tokens on. index.html runs the same resolution before first paint.

import { useEffect, useState } from 'react'

export type ThemePref = 'system' | 'light' | 'dark'

const KEY = 'dhaga.theme'
const media = () => window.matchMedia('(prefers-color-scheme: dark)')

function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY)
    if (v === 'light' || v === 'dark' || v === 'system') return v
  } catch {
    /* storage blocked: fall back to the system setting */
  }
  return 'system'
}

export function resolveTheme(pref: ThemePref): 'light' | 'dark' {
  if (pref === 'system') return media().matches ? 'dark' : 'light'
  return pref
}

export function useTheme(): [ThemePref, (p: ThemePref) => void] {
  const [pref, setPref] = useState<ThemePref>(readPref)

  useEffect(() => {
    const apply = () => {
      document.documentElement.dataset.theme = resolveTheme(pref)
    }
    apply()
    try {
      localStorage.setItem(KEY, pref)
    } catch {
      /* the choice just won't persist */
    }
    if (pref !== 'system') return
    // Follow the OS live while on "System".
    const mq = media()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [pref])

  return [pref, setPref]
}
