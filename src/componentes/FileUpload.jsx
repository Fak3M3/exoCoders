import { useState } from 'react'

export const FileUpload = () => {
  const [file, setFile] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0]
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile)
      setError(null)
    } else {
      setError('Por favor selecciona un archivo CSV válido')
      setFile(null)
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!file) {
      setError('Por favor selecciona un archivo')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      })

      const result = await response.json()

      if (result.success) {
        setPrediction(result.prediction)
      } else {
        setError(result.error || 'Error en la predicción')
      }
    } catch {
      setError('Error de conexión con el servidor')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-md mx-auto p-6 bg-[#3c3e44] rounded-lg shadow-lg border border-gray-600">
      <h2 className="text-2xl font-bold mb-4 text-[#c0df40] text-center">
        Subir CSV para Predicción
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-2">
            Seleccionar archivo CSV
          </label>
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-[#c0df40] file:text-[#272b33] hover:file:bg-[#a8c030]"
          />
        </div>

        {file && (
          <div className="text-sm text-green-400">
            Archivo seleccionado: {file.name}
          </div>
        )}

        {error && (
          <div className="text-sm text-red-400 bg-red-900/30 p-2 rounded">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={!file || loading}
          className="w-full bg-[#c0df40] text-[#272b33] py-2 px-4 rounded-md font-bold hover:bg-[#a8c030] disabled:bg-gray-500 disabled:text-gray-300"
        >
          {loading ? 'Procesando...' : 'Hacer Predicción'}
        </button>
      </form>

      {prediction && (
        <div className="mt-6 p-4 bg-[#272b33] rounded-md">
          <h3 className="font-semibold text-[#c0df40]">Resultado:</h3>
          <p className="text-gray-200">{JSON.stringify(prediction, null, 2)}</p>
        </div>
      )}
    </div>
  )
}

export default FileUpload