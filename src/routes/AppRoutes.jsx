import { Route, Routes } from "react-router-dom";
import { Home } from "../pages/Home";
import { ExoPlanetas } from "../pages/ExoPlanetas";


const AppRoutes = () => {
    return (
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/Busqueda" element={<ExoPlanetas />} />
        </Routes>
    )
}
export default AppRoutes;