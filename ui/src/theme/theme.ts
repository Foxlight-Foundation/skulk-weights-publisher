/**
 * Theme palette for skulk-ui.
 *
 * Tokens copied verbatim from Skulk dashboard-react. skulk-ui uses light mode
 * only — no theme toggle in V1.
 */

const sharedFonts = {
  body: "'Outfit', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  mono: "'JetBrains Mono', 'Fira Code', monospace",
} as const;

const sharedFontSizes = {
  xs: '12px',
  sm: '14px',
  md: '16px',
  lg: '18px',
  xl: '22px',
  xxl: '30px',
  label: '13px',
} as const;

const sharedRadii = {
  sm: '4px',
  md: '8px',
  lg: '12px',
  xl: '16px',
} as const;

const sharedSpacing = {
  xs: '4px',
  sm: '8px',
  md: '16px',
  lg: '24px',
  xl: '32px',
} as const;

const lightColors = {
  bg: '#eef3fb',
  bgGradient: `
    radial-gradient(ellipse at 0% 0%, #dbeafe 0%, transparent 50%),
    radial-gradient(ellipse at 100% 100%, #e0e7ff 0%, transparent 50%),
    #eef3fb
  `,
  surface: '#ffffff',
  surfaceHover: '#e6edf8',
  surfaceElevated: 'rgba(255, 255, 255, 0.96)',
  surfaceSunken: 'rgba(15, 23, 42, 0.04)',
  header: 'rgba(255, 255, 255, 0.78)',
  headerBorder: 'linear-gradient(to right, rgba(30, 64, 175, 0.18), rgba(30, 64, 175, 0.03))',
  overlay: 'rgba(15, 23, 42, 0.42)',
  shadow: 'rgba(15, 23, 42, 0.10)',
  shadowStrong: 'rgba(15, 23, 42, 0.18)',

  border: 'rgba(30, 64, 175, 0.16)',
  borderLight: 'rgba(30, 64, 175, 0.10)',
  borderStrong: 'rgba(30, 64, 175, 0.32)',

  text: '#0f172a',
  textSecondary: 'rgba(15, 23, 42, 0.72)',
  textMuted: 'rgba(15, 23, 42, 0.5)',
  textOnAccent: '#ffffff',

  // In light mode gold maps to deep blue — the brand accent.
  gold: '#1d4ed8',
  goldDim: 'rgba(29, 78, 216, 0.55)',
  goldBg: 'rgba(29, 78, 216, 0.10)',
  goldStrong: '#1e3a8a',

  accent: '#0ea5e9',
  accentHover: '#0284c7',
  accentBg: 'rgba(14, 165, 233, 0.12)',

  error: '#dc2626',
  errorBg: 'rgba(220, 38, 38, 0.10)',
  errorText: '#991b1b',
  errorOnSurface: '#b91c1c',

  warning: '#475569',
  warningBg: 'rgba(71, 85, 105, 0.08)',
  warningText: '#1e293b',
  warningOnSurface: '#b45309',

  info: '#1d4ed8',
  infoBg: 'rgba(29, 78, 216, 0.10)',

  healthy: '#0ea5e9',
  unhealthy: '#dc2626',

  bgMeshLine: 'rgba(29, 78, 216, 0.16)',
  bgMeshNode: 'rgba(29, 78, 216, 0.12)',
} as const;

function buildTheme(colors: typeof lightColors) {
  return {
    colors,
    fonts: sharedFonts,
    fontSizes: sharedFontSizes,
    radii: sharedRadii,
    spacing: sharedSpacing,
  } as const;
}

export const lightTheme = buildTheme(lightColors);

export type Theme = typeof lightTheme;
