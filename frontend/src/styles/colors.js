/**
 * Color Scheme Configuration
 * 
 * This file defines the color palette for the Clinical Agent frontend.
 * Modify these values to change the app's color scheme.
 * 
 * Color Palette:
 * - Primary (Pink): #f97fbe - Main accent color for buttons, highlights
 * - Dark: #181818 - Text and dark elements
 * - Muted (Gray): #898989 - Secondary text, borders
 * - Light/Background: #f0f0f0 - Background color
 */

const colors = {
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

  // Dark color for text
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

  // Background and light surfaces
  background: {
    DEFAULT: '#f0f0f0',
    light: '#ffffff',
    dark: '#e5e5e5',
  },

  // Light color for cards, messages
  light: {
    DEFAULT: '#f0f0f0',
    50: '#ffffff',
    100: '#fafafa',
    200: '#f5f5f5',
    300: '#f0f0f0',
    400: '#e5e5e5',
    500: '#d4d4d4',
  },
};

export default colors;

/**
 * CSS Custom Properties (for use in CSS files)
 * 
 * Add these to your index.css or global styles:
 * 
 * :root {
 *   --color-primary: #f97fbe;
 *   --color-primary-light: #fba8d0;
 *   --color-primary-dark: #e75a9c;
 *   --color-dark: #181818;
 *   --color-muted: #898989;
 *   --color-background: #f0f0f0;
 *   --color-light: #ffffff;
 * }
 */
