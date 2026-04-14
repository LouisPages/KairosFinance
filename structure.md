# PE25 — Kairos Finance

Application web de gestion et simulation de portefeuille boursier (projet PE 25, École Centrale Lyon). Elle permet de composer un portefeuille à partir d’actions du NASDAQ, Dow Jones et S&P 500, de visualiser des performances et d’utiliser des modèles d’optimisation type Markowitz.

Ce README décrit comment le projet est organisé et à quoi sert chaque type de fichier, pour des personnes qui découvrent React, Node.js et TypeScript.

---

## Contexte technique en bref

Le projet repose sur **Node.js** (environnement qui exécute du JavaScript en dehors du navigateur et gère les outils du projet), **TypeScript** (JavaScript avec des types pour mieux détecter les erreurs), **React** (bibliothèque pour construire l’interface en composants réutilisables) et **Vite** (outil qui assemble et sert l’application en développement et qui la compile pour la production). Quand vous lancez `npm run dev`, Node exécute Vite, qui lit votre code, le transforme et l’envoie au navigateur.

---

## Point d’entrée : du fichier HTML au composant principal

Tout part d’un fichier HTML à la racine du projet : **`index.html`**. C’est la seule page HTML du site. Elle contient une balise `<div id="root">` vide et une référence au fichier **`src/main.tsx`**. Le navigateur charge cette page, puis exécute le script `main.tsx`.

Le fichier **`src/main.tsx`** est le point d’entrée du code. Il importe le fichier de styles globaux **`src/index.css`** et le composant **`App`** défini dans **`src/App.tsx`**. Il demande à React de « rendre » ce composant à l’intérieur de la div `root`. Concrètement, tout ce que vous voyez à l’écran est produit par ce composant `App` et les composants qu’il inclut.

---

## Le composant App et la structure de l’application

**`src/App.tsx`** est le composant racine. Il enveloppe toute l’application avec des « fournisseurs » (providers) qui donnent des capacités à tous les composants enfants : gestion des requêtes asynchrones (React Query), des infobulles (Tooltip), des notifications (Toaster). Il contient aussi le routeur (**React Router**) : selon l’URL (par exemple `/`, `/portfolio`, `/simulation`, `/architecture`), un composant de page différent s’affiche. La barre de navigation (**Navbar**) et le pied de page (**Footer**) entourent la zone où ces pages s’affichent. Un petit composant **ScrollToTop** remet le défilement en haut à chaque changement de page.

Les routes sont définies ainsi : la page d’accueil correspond à **`src/pages/Home.tsx`**, « Mon Portefeuille » à **`src/pages/Portfolio.tsx`**, « Simulation » à **`src/pages/Simulation.tsx`**, « Architecture » à **`src/pages/Architecture.tsx`**. Toute autre URL mène à **`src/pages/NotFound.tsx`**. Le fichier **`src/pages/Index.tsx`** existe dans le projet mais n’est pas utilisé par les routes actuelles ; c’est une page de secours ou de démarrage que vous pouvez réutiliser ou modifier.

---

## Les pages et les composants

Le dossier **`src/pages`** contient les écrans principaux. Chaque fichier exporte un composant qui représente une page complète. Ces pages utilisent des composants plus petits (boutons, cartes, graphiques, etc.) pour construire l’interface.

Le dossier **`src/components`** regroupe les blocs réutilisables. **Navbar.tsx** et **Footer.tsx** sont la barre du haut et le pied de page. **NavLink.tsx** sert à afficher des liens de navigation. **ScrollToTop.tsx** gère le scroll en haut à chaque changement de route.

Le sous-dossier **`src/components/ui`** contient des composants d’interface génériques (boutons, champs de formulaire, cartes, dialogues, onglets, tableaux, etc.). Ils viennent de la bibliothèque **shadcn/ui** (déclarée dans **`components.json`** à la racine) et sont personnalisables ; ils constituent la « boîte à outils » visuelle du projet.

---

## Données, logique et styles

Les données fictives des actions (symboles, prix, indices, indicateurs) sont dans **`src/data/mockStocks.ts`**. Ce fichier exporte une liste d’objets et des types TypeScript (par exemple `Stock`) ainsi que des fonctions utilitaires comme `generateHistoricalData`. Les pages **Portfolio** et **Simulation** importent ces données pour afficher les actions et les graphiques.

Le dossier **`src/lib`** contient des utilitaires partagés. **`utils.ts`** fournit notamment la fonction **`cn`** qui combine des noms de classes CSS (avec **Tailwind**) de façon propre ; elle est utilisée dans beaucoup de composants pour le style conditionnel.

Le dossier **`src/hooks`** contient des « hooks » React réutilisables. Par exemple **`use-mobile.tsx`** détecte si l’écran est en mode mobile pour adapter l’interface ; **`use-toast.ts`** est lié au système de notifications. Les hooks permettent de factoriser la logique (état, effets) entre plusieurs composants.

Les styles globaux sont dans **`src/index.css`**. Il importe des polices, les bases de **Tailwind CSS** et définit des variables CSS (couleurs, thème) utilisées dans tout le projet. **`src/App.css`** existe mais peut être peu ou pas utilisé selon les versions ; les styles sont en grande partie gérés par Tailwind et les composants.

---

## Fichiers de configuration à la racine

**`package.json`** décrit le projet et ses dépendances. Les « scripts » indiquent les commandes à lancer : `npm run dev` démarre le serveur de développement (Vite), `npm run build` produit la version optimisée pour la mise en ligne, `npm run preview` sert cette version buildée localement, `npm run test` lance les tests, `npm run lint` vérifie le code avec ESLint.

**`vite.config.ts`** configure Vite : le plugin React pour compiler les fichiers `.tsx`, le port du serveur, et surtout l’**alias** `@` qui pointe vers le dossier **`src`**. C’est pour cela qu’on peut écrire `import { Button } from "@/components/ui/button"` au lieu de chemins relatifs comme `../../components/ui/button`.

**`tsconfig.json`** et **`tsconfig.app.json`** (ainsi que **`tsconfig.node.json`** pour la partie Node) définissent comment TypeScript lit le projet : options du langage, où chercher les fichiers, et le fait que `@/*` correspond à `./src/*`. **`tailwind.config.ts`** et **`postcss.config.js`** configurent Tailwind et PostCSS pour générer les classes CSS utilisées dans les composants.

**`components.json`** est utilisé par l’outil shadcn/ui pour savoir où installer les composants (dossier `src/components/ui`) et comment ils s’intègrent avec Tailwind et les alias. **`eslint.config.js`** et **`vitest.config.ts`** configurent respectivement le linter et le lanceur de tests.

---

## Tests et types

Le dossier **`src/test`** contient la configuration et des tests. **`setup.ts`** prépare l’environnement de test (par exemple en simulant certaines APIs du navigateur). **`example.test.ts`** est un exemple de fichier de test. Les tests vérifient que des parties du code se comportent comme prévu sans lancer toute l’application dans le navigateur.

Le fichier **`src/vite-env.d.ts`** indique à TypeScript que le projet utilise l’environnement Vite (par exemple pour reconnaître l’import de fichiers ou les variables d’environnement spécifiques à Vite).

---

## Résumé du flux

Quand vous ouvrez l’application dans le navigateur, le serveur (ou le build) envoie **index.html**. Le navigateur charge **main.tsx**, qui applique les styles de **index.css** et affiche le composant **App**. **App** configure le routeur et affiche la **Navbar**, le **Footer** et, selon l’URL, l’une des pages (**Home**, **Portfolio**, **Simulation**, **Architecture** ou **NotFound**). Ces pages s’appuient sur les composants de **components** et **components/ui**, sur les données de **data/mockStocks.ts**, sur les utilitaires de **lib** et les hooks de **hooks**. Les fichiers à la racine (package.json, vite.config.ts, tsconfig, tailwind, etc.) définissent comment tout cela est installé, compilé et exécuté.

Pour modifier le contenu d’une page, éditez le fichier correspondant dans **`src/pages`**. Pour changer l’en-tête ou le pied de page, modifiez **Navbar.tsx** et **Footer.tsx**. Pour ajouter une nouvelle route, ajoutez une ligne dans les **Routes** de **App.tsx** et créez un nouveau fichier dans **pages** si besoin.
