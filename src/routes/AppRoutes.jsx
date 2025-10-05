import { Route, Routes } from "react-router-dom";
import { Home } from "../pages/Home";
import { ExoPlanetas } from "../pages/ExoPlanetas";
import { NotFound } from "../pages/NotFound";


const AppRoutes = () => {
    return (
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/Busqueda" element={<ExoPlanetas />} />
            <Route path='*' element={<NotFound />} />
        </Routes>
    )
}
export default AppRoutes;