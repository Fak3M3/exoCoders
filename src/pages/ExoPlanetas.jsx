import React, { useCallback } from 'react'
import Particles from 'react-tsparticles'
import FileUpload from '../componentes/FileUpload'

export const ExoPlanetas = () => {
  const particlesInit = useCallback(async () => {}, [])

  return (
    <div className="relative min-h-screen flex items-center justify-center">
      {/* Fondo animado tipo cinturón de meteoritos */}
      <Particles
        id="meteoritos"
        init={particlesInit}
        options={{
          fullScreen: { enable: true, zIndex: -1 },
          particles: {
            number: { value: 80 },
            color: { value: ['#ffffff', '#a8a8a8', '#87ceeb'] },
            shape: { type: 'circle' },
            opacity: { value: 0.8 },
            size: { value: { min: 1, max: 4 } },
            move: {
              enable: true,
              speed: 3,
              direction: 'bottom-right',
              straight: true,
              outModes: { default: 'out' }
            }
          },
          background: { color: '#000000' }
        }}
      />

      {/* Componente FileUpload centrado con transparencia */}
      <div className="max-w-lg w-full backdrop-blur-md bg-black/30 p-6 rounded-lg shadow-lg border border-gray-700">
        <FileUpload />
      </div>
    </div>
  )
}