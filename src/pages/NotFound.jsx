import React from "react";
import { Link } from "react-router-dom";
import image404 from "../assets/404_NotFoundImage.png";

export const NotFound = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-black text-white text-center px-4">
      <h1 className="text-5xl font-extrabold text-[#c0df40] mb-4">404</h1>
      <h2 className="text-2xl font-bold mb-2">Página no encontrada</h2>
      <p className="text-gray-300 mb-6">¡¡La exploración se ha detenido!!</p>
      <img
        src={image404}
        alt="Página no encontrada"
        className="w-64 h-auto mb-6 rounded-lg shadow-lg"
      />
      <Link
        to="/"
        className="bg-[#c0df40] text-[#272b33] font-bold py-2 px-6 rounded-lg hover:bg-[#a8c030] transition duration-300"
      >
        Volver al Inicio
      </Link>
    </div>
  );
};