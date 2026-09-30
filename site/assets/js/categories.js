// Informasi per kategori sampah: nama Indonesia, ikon, contoh barang, dan cara membuang yang benar.
// Warna diambil dari CSS variable --c-<kelas> supaya ikut tema terang/gelap.

const svg = (paths) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths}</svg>`;

export const CATEGORIES = {
  cardboard: {
    name: "Kardus",
    en: "Cardboard",
    icon: svg('<path d="M12 3 3.5 7.5v9L12 21l8.5-4.5v-9L12 3Z"/><path d="M3.5 7.5 12 12l8.5-4.5M12 12v9"/>'),
    bin: "Tempat sampah kertas (umumnya biru) atau bank sampah",
    examples: ["Kardus paket dan kotak kemasan", "Kotak sepatu, kotak mi instan", "Karton susu kering (tanpa lapisan plastik)"],
    tips: [
      "Kosongkan isinya dan lepas selotip atau label plastik.",
      "Lipat atau pipihkan supaya hemat tempat.",
      "Jaga tetap kering. Kardus basah atau berminyak sulit didaur ulang.",
    ],
  },
  glass: {
    name: "Kaca",
    en: "Glass",
    icon: svg('<path d="M10 2h4v4.2c0 .6.3 1.1.7 1.5A4.5 4.5 0 0 1 16 11v9a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-9a4.5 4.5 0 0 1 1.3-3.3c.4-.4.7-.9.7-1.5V2Z"/><path d="M8 13h8"/>'),
    bin: "Tempat sampah anorganik (umumnya kuning) atau bank sampah",
    examples: ["Botol kecap, sirup, dan minuman", "Toples selai atau sambal", "Botol obat dan kosmetik yang sudah kosong"],
    tips: [
      "Kosongkan dan bilas sisa isinya.",
      "Lepas tutupnya. Tutup logam masuk ke kategori logam.",
      "Kaca pecah dibungkus kertas tebal dan diberi tanda supaya petugas tidak terluka.",
    ],
  },
  metal: {
    name: "Logam",
    en: "Metal",
    icon: svg('<ellipse cx="12" cy="5" rx="6" ry="2.2"/><path d="M6 5v14c0 1.2 2.7 2.2 6 2.2s6-1 6-2.2V5"/><path d="M6 9.5c0 1.2 2.7 2.2 6 2.2s6-1 6-2.2"/>'),
    bin: "Tempat sampah anorganik (umumnya kuning) atau pengepul logam",
    examples: ["Kaleng minuman dan kaleng sarden", "Kaleng susu kental manis", "Tutup botol dan aluminium foil yang bersih"],
    tips: [
      "Bilas sisa makanan atau minuman supaya tidak bau.",
      "Pipihkan kaleng aluminium untuk menghemat tempat.",
      "Kaleng semprot (aerosol) yang masih berisi termasuk sampah B3, jangan dilubangi.",
    ],
  },
  paper: {
    name: "Kertas",
    en: "Paper",
    icon: svg('<path d="M6 2h8l4 4v16H6V2Z"/><path d="M14 2v4h4M9 11h6M9 15h6M9 19h4"/>'),
    bin: "Tempat sampah kertas (umumnya biru) atau bank sampah",
    examples: ["Koran, majalah, dan brosur", "Kertas HVS, buku tulis bekas", "Amplop dan kertas kado tanpa plastik"],
    tips: [
      "Pastikan kering dan bersih.",
      "Kertas berminyak (bungkus gorengan) dan tisu bekas tidak bisa didaur ulang, buang ke residu.",
      "Lepas klip, staples, atau jendela plastik pada amplop kalau memungkinkan.",
    ],
  },
  plastic: {
    name: "Plastik",
    en: "Plastic",
    icon: svg('<path d="M10 2h4v2.5l1.5 2A3 3 0 0 1 16 8v12a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2V8c0-.6.2-1.1.5-1.5L10 4.5V2Z"/><path d="M8 11h8M8 15h8"/>'),
    bin: "Tempat sampah anorganik (umumnya kuning) atau bank sampah",
    examples: ["Botol air mineral dan galon", "Botol sampo dan sabun cair", "Gelas plastik dan wadah makanan"],
    tips: [
      "Kosongkan dan bilas, lalu pipihkan botol.",
      "Cek kode segitiga di bawah kemasan. Kode 1 (PET) dan 2 (HDPE) paling mudah didaur ulang.",
      "Kantong kresek dan sachet sering tidak diterima, tanyakan dulu ke bank sampah setempat.",
    ],
  },
  trash: {
    name: "Residu",
    en: "Trash",
    icon: svg('<path d="M4 6h16M9 6V4h6v2M6 6l1 15h10l1-15"/><path d="M10 10v7M14 10v7"/>'),
    bin: "Tempat sampah residu (umumnya abu-abu)",
    examples: ["Bungkus snack berlapis aluminium", "Sachet kopi atau sampo, sedotan", "Tisu bekas, styrofoam kotor, popok"],
    tips: [
      "Buang ke tempat sampah residu, karena sulit atau tidak bisa didaur ulang.",
      "Cara terbaik adalah mengurangi pemakaiannya, misalnya membawa botol minum sendiri.",
      "Sebagian plastik kemasan bersih bisa dijadikan ecobrick.",
    ],
  },
};
