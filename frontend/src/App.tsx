import './App.css';
import './index.css';
import '../app/globals.css';
import { ThemeProvider } from "@/components/theme-provider"
import { Routes, Route, BrowserRouter } from "react-router-dom";
import {Home} from "@/pages/home-page"
import { NoMatch } from '@/pages/not-found';
import { Dashboard } from './pages/dashboard-page';
import { Layout } from '@/layouts/layout';
import { TooltipProvider } from "@/components/ui/tooltip";
function App() {

  return (
    <>
      <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <TooltipProvider>

      <BrowserRouter>

      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="*" element={<NoMatch />} />
        </Route>
      </Routes>
      </BrowserRouter>
      </TooltipProvider>
      </ThemeProvider>

    </>
  );
}

export default App;

