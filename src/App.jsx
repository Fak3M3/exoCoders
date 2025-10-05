import React from 'react'
import './App.css'
import { BrowserRouter } from 'react-router-dom'
import { Navbar } from './componentes/Navbar'
import AppRoutes from './routes/AppRoutes'


function App() {
  
  return (
    <>
      <BrowserRouter>
        <Navbar />
        <AppRoutes />
      </BrowserRouter>
    </>
  )
}

export default App
