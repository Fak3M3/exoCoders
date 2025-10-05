import { NavLink } from "react-router-dom";

export const Navbar = () =>{ 
    return (
        <nav className=" bg-[#3c3e44] p-4 mb-0 rounded-lg flex justify-between items-center">
            <ul className="flex gap-8 list-none m-0 p-0">
                <li>
                    <NavLink 
                        to="/"
                        className={({ isActive }) =>
                            `text-[1.2rem] font-bold px-4 py-2 rounded transition-all duration-300 ${
                            isActive
                                ? "bg-[#c0df40] text-[#272b33]"
                                : "text-[#f5f5f5] hover:bg-[#555] hover:text-white"
                            }`
                        }>Home
                    </NavLink>
                </li>
                <li>
                    <NavLink
                        to="/Busqueda"
                        className={({ isActive }) => `text-[1.2rem] font-bold px-4 py-2 rounded transition-all duration-300 ${
                            isActive
                                ? "bg-[#c0df40] text-[#272b33]"
                                : "text-[#f5f5f5] hover:bg-[#555] hover:text-white"
                            }`
                        }
                    >
                        Busqueda
                    </NavLink>
                </li>
            </ul>
        </nav>
     )
}