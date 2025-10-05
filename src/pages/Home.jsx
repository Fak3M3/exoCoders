import React, { useCallback, useState } from "react";
import Particles from "react-tsparticles";
import { loadStarsPreset } from "tsparticles-preset-stars";

export const Home = () => {
  const particlesInicio = useCallback(async (engine) => {
    await loadStarsPreset(engine);
  }, []);

  const [selectedCategory, setSelectedCategory] = useState("Tipos");

  const categories = {
    Tipos: [
      {
        name: "Júpiter caliente",
        info: "Gigantes gaseosos abrasadoramente cerca de su estrella.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/1234_hotjupiter.jpg"
      },
      {
        name: "Supertierras",
        info: "Planetas rocosos más grandes que la Tierra, posibles océanos.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/5678_superearth.jpg"
      },
      {
        name: "Mini-Neptunos",
        info: "Pequeños gigantes con atmósferas densas de hidrógeno y helio.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/91011_minineptune.jpg"
      },
      {
        name: "Errantes",
        info: "Exoplanetas que no orbitan ninguna estrella.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/121314_rogueplanet.jpg"
      }
    ],
    Métodos: [
      {
        name: "Tránsito",
        info: "Detecta la disminución de luz cuando el planeta pasa frente a su estrella.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/5678_transit.jpg"
      },
      {
        name: "Velocidad radial",
        info: "Mide el bamboleo de la estrella causado por la gravedad del planeta.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/91011_radialvelocity.jpg"
      },
      {
        name: "Imagen directa",
        info: "Captura imágenes del planeta bloqueando la luz de la estrella.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/121314_directimage.jpg"
      }
    ],
    Misiones: [
      {
        name: "Kepler",
        info: "Descubrió miles de exoplanetas usando el método de tránsito.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/kepler.jpg"
      },
      {
        name: "TESS",
        info: "Busca exoplanetas cercanos a estrellas brillantes.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/tess.jpg"
      },
      {
        name: "JWST",
        info: "Analiza atmósferas para buscar signos de habitabilidad.",
        img: "https://exoplanets.nasa.gov/system/resources/detail_files/jwst.jpg"
      }
    ]
  };

  return (
    <div className="relative min-h-screen w-screen text-white">
      {/* Fondo global */}
      <Particles
        init={particlesInicio}
        options={{
          preset: "stars",
          fullScreen: { enable: true, zIndex: -1 }
        }}
      />

      {/* Layout principal */}
      <div className="flex flex-col md:flex-row min-h-screen">
        {/* Sidebar */}
        <aside className="w-full md:w-1/4 p-6 border-b md:border-b-0 md:border-r border-gray-700 bg-black/30 backdrop-blur-sm">
          <h2 className="text-2xl font-bold mb-6">Categorías</h2>
          <ul className="space-y-4">
            {Object.keys(categories).map((cat) => (
              <li
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`cursor-pointer px-4 py-2 rounded-lg transition ${
                  selectedCategory === cat ? "bg-[#c0df40] text-black" : "hover:bg-gray-700/50"
                }`}
              >
                {cat}
              </li>
            ))}
          </ul>
        </aside>

        {/* Main content */}
        <main className="flex-1 p-6 overflow-y-scroll h-screen">
          <h1 className="text-4xl font-extrabold mb-6">Explora lo desconocido</h1>
          <p className="text-gray-300 mb-8">
            Descubre mundos más allá de nuestro sistema solar. Aprende sobre los diferentes tipos de exoplanetas,
            métodos de detección y misiones que los estudian.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {categories[selectedCategory].map((item) => (
              <div
                key={item.name}
                className="rounded-lg overflow-hidden border border-gray-700 bg-black/30 backdrop-blur-sm hover:scale-105 transition transform"
              >
                <img src={item.img} alt={item.name} className="w-full h-40 object-cover" />
                <div className="p-4">
                  <h3 className="text-xl font-bold mb-2">{item.name}</h3>
                  <p className="text-gray-300">{item.info}</p>
                </div>
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
};