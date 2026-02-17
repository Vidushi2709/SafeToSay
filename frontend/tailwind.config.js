/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary accent color (pink)
        primary: {
          DEFAULT: '#f97fbe',
          light: '#fba8d0',
          dark: '#e75a9c',
          50: '#fef1f7',
          100: '#fde6f1',
          200: '#fccce3',
          300: '#fba8d0',
          400: '#f97fbe',
          500: '#f054a0',
          600: '#e03380',
          700: '#c22565',
          800: '#a02254',
          900: '#852148',
        },
        // Dark color for text - changed to pure black
        dark: {
          DEFAULT: '#181818',
          50: '#f5f5f5',
          100: '#e0e0e0',
          200: '#b8b8b8',
          300: '#8f8f8f',
          400: '#666666',
          500: '#4d4d4d',
          600: '#333333',
          700: '#262626',
          800: '#1a1a1a',
          900: '#181818',
        },
        // Muted gray for secondary elements
        muted: {
          DEFAULT: '#898989',
          light: '#a8a8a8',
          dark: '#6a6a6a',
        },
        // Background color
        background: {
          DEFAULT: '#f0f0f0',
          light: '#ffffff',
          dark: '#e5e5e5',
        },
        // Light color for cards
        light: {
          DEFAULT: '#f0f0f0',
          50: '#ffffff',
          100: '#fafafa',
          200: '#f5f5f5',
          300: '#f0f0f0',
          400: '#e5e5e5',
          500: '#d4d4d4',
        },
      },
    },
  },
  plugins: [],
}