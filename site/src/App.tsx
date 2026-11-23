import { Navbar } from "./components/Navbar.tsx";
import { Hero } from "./components/Hero.tsx";
import { Results } from "./components/Results.tsx";
import { Flash } from "./components/Flash.tsx";
import { LongContext } from "./components/LongContext.tsx";
import { Models } from "./components/Models.tsx";
import { GetStarted } from "./components/GetStarted.tsx";
import { Footer } from "./components/Footer.tsx";

export default function App() {
  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <Navbar />
      <main id="main">
        <Hero />
        <Results />
        <Flash />
        <LongContext />
        <Models />
        <GetStarted />
      </main>
      <Footer />
    </>
  );
}
