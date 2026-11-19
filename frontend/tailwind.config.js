/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        luxury: {
          50: '#F9F8F6',
          100: '#F2EFEB',
          200: '#E5DFD7',
          300: '#D4C7B8',
          400: '#BAA38A',
          500: '#9E8262',
          600: '#84694D',
          700: '#69513B',
          800: '#4F3C2C',
          900: '#2A1F16',
          950: '#140E0A',
        },
        surface: {
          base: '#0B0B0B',
          card: '#111111',
          subtle: '#171717',
          border: '#27272A',
          hover: '#1F1F23',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        serif: ['Playfair Display', 'Georgia', 'serif'],
      }
    },
  },
  plugins: [],
}
