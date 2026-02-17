/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
      colors: {
        // Primary — Teal (clinical, trustworthy)
        primary: {
          DEFAULT: '#0d9488',
          light: '#2dd4bf',
          dark: '#0f766e',
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        },
        // Accent — pink (sparingly for highlights)
        accent: {
          DEFAULT: '#f97fbe',
          light: '#fba8d0',
          dark: '#e75a9c',
        },
        // Dark — slate for text
        dark: {
          DEFAULT: '#0f172a',
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        // Muted
        muted: {
          DEFAULT: '#64748b',
          light: '#94a3b8',
          dark: '#475569',
        },
        // Background
        background: {
          DEFAULT: '#f8fafc',
          light: '#ffffff',
          dark: '#f1f5f9',
        },
        // Light surfaces
        light: {
          DEFAULT: '#f1f5f9',
          50: '#ffffff',
          100: '#f8fafc',
          200: '#f1f5f9',
          300: '#e2e8f0',
          400: '#cbd5e1',
          500: '#94a3b8',
        },
      },
    },
  },
  plugins: [],
}