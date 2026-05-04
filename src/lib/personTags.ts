/** Prénoms proposés à l'enregistrement ; ordre = palette d'index stable. */
export const PERSON_TAGS = ["Syuma", "Akram", "Louis", "Bruno", "Aurélien", "Augustin"] as const;

export type PersonTagId = (typeof PERSON_TAGS)[number];

const TEST_TAG_GREY_PILL = "bg-slate-500/15 text-slate-700 dark:text-slate-400";
const TEST_TAG_GREY_DOT = "bg-slate-500";

const PILL_BY_NAME: Record<string, string> = {
  Syuma: "bg-sky-500/15 text-sky-700 dark:text-sky-400",
  Akram: "bg-violet-500/15 text-violet-700 dark:text-violet-400",
  Louis: "bg-rose-500/15 text-rose-700 dark:text-rose-400",
  Bruno: "bg-amber-500/15 text-amber-800 dark:text-amber-400",
  Aurélien: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  Augustin: "bg-cyan-500/15 text-cyan-700 dark:text-cyan-400",
  "Simulation de Test": TEST_TAG_GREY_PILL,
  "test système simulation": TEST_TAG_GREY_PILL,
};

const DOT_BY_NAME: Record<string, string> = {
  Syuma: "bg-sky-500",
  Akram: "bg-violet-500",
  Louis: "bg-rose-500",
  Bruno: "bg-amber-500",
  Aurélien: "bg-emerald-500",
  Augustin: "bg-cyan-500",
  "Simulation de Test": TEST_TAG_GREY_DOT,
  "test système simulation": TEST_TAG_GREY_DOT,
};

const FALLBACK_PILLS = [
  "bg-fuchsia-500/15 text-fuchsia-700 dark:text-fuchsia-400",
  "bg-indigo-500/15 text-indigo-700 dark:text-indigo-400",
  "bg-teal-500/15 text-teal-700 dark:text-teal-400",
  "bg-lime-500/20 text-lime-800 dark:text-lime-400",
] as const;

const FALLBACK_DOTS = ["bg-fuchsia-500", "bg-indigo-500", "bg-teal-500", "bg-lime-500"] as const;

function tagBucket(tag: string): number {
  let h = 0;
  for (let i = 0; i < tag.length; i++) h = (Math.imul(31, h) + tag.charCodeAt(i)) | 0;
  return Math.abs(h);
}

/** Classes Tailwind (fond + texte) pour un pastille de prénom. */
export function getPersonTagPillClass(tag: string): string {
  return PILL_BY_NAME[tag] ?? FALLBACK_PILLS[tagBucket(tag) % FALLBACK_PILLS.length];
}

/** Pastille pleine (sélecteurs, légendes). */
export function getPersonTagDotClass(tag: string): string {
  return DOT_BY_NAME[tag] ?? FALLBACK_DOTS[tagBucket(tag) % FALLBACK_DOTS.length];
}
