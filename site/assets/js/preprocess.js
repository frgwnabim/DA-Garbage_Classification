// Preprocessing yang meniru torchvision + PIL persis seperti saat training:
//   Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(mean, std)
// Resize memakai algoritma resample bilinear milik PIL (dengan antialias dan bobot fixed-point 22 bit),
// jadi untuk gambar yang sama hasilnya identik dengan notebook, bukan sekadar mirip.

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;
const HALF = 2 ** (PRECISION_BITS - 1);

// Filter segitiga (bilinear) milik PIL, support = 1
const bilinear = (x) => {
  x = Math.abs(x);
  return x < 1 ? 1 - x : 0;
};

// Salinan precompute_coeffs + normalize_coeffs_8bpc di Pillow (src/libImaging/Resample.c)
function computeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const coeffs = new Float64Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    const k = new Float64Array(xmax);
    for (let x = 0; x < xmax; x++) {
      const w = bilinear((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      coeffs[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, coeffs, ksize };
}

const clip8 = (v) => {
  const r = Math.floor(v / ONE);
  return r < 0 ? 0 : r > 255 ? 255 : r;
};

// Resize RGB (Uint8, 3 channel) memakai dua pass seperti PIL: horizontal dulu, lalu vertikal
function resizeRGB(src, w, h, outW, outH) {
  let data = src;
  let curW = w;
  if (outW !== w) {
    const { bounds, coeffs, ksize } = computeCoeffs(w, outW);
    const tmp = new Uint8ClampedArray(outW * h * 3);
    for (let y = 0; y < h; y++) {
      const row = y * w * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const kOff = xx * ksize;
        let r = HALF, g = HALF, b = HALF;
        for (let x = 0; x < xmax; x++) {
          const k = coeffs[kOff + x];
          const p = row + (x + xmin) * 3;
          r += data[p] * k;
          g += data[p + 1] * k;
          b += data[p + 2] * k;
        }
        const o = (y * outW + xx) * 3;
        tmp[o] = clip8(r);
        tmp[o + 1] = clip8(g);
        tmp[o + 2] = clip8(b);
      }
    }
    data = tmp;
    curW = outW;
  }
  if (outH !== h) {
    const { bounds, coeffs, ksize } = computeCoeffs(h, outH);
    const out = new Uint8ClampedArray(curW * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      const kOff = yy * ksize;
      for (let x = 0; x < curW; x++) {
        let r = HALF, g = HALF, b = HALF;
        for (let y = 0; y < ymax; y++) {
          const k = coeffs[kOff + y];
          const p = ((y + ymin) * curW + x) * 3;
          r += data[p] * k;
          g += data[p + 1] * k;
          b += data[p + 2] * k;
        }
        const o = (yy * curW + x) * 3;
        out[o] = clip8(r);
        out[o + 1] = clip8(g);
        out[o + 2] = clip8(b);
      }
    }
    data = out;
  }
  return data;
}

// round() di Python membulatkan .5 ke bilangan genap (banker's rounding), dipakai torchvision CenterCrop
function roundHalfEven(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// Ukuran hasil Resize(256) torchvision: sisi pendek jadi 256, sisi panjang int(256 * panjang / pendek)
export function resizedSize(w, h, shorter) {
  if (w <= h) return [shorter, Math.trunc((shorter * h) / w)];
  return [Math.trunc((shorter * w) / h), shorter];
}

/**
 * Ubah ImageData (RGBA) menjadi Float32Array NCHW [1, 3, crop, crop] sesuai preprocessing.json.
 */
export function preprocess(imageData, config) {
  const { width: w, height: h, data: rgba } = imageData;
  const rgb = new Uint8ClampedArray(w * h * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  const [rw, rh] = resizedSize(w, h, config.resize.shorter_side);
  const resized = resizeRGB(rgb, w, h, rw, rh);

  const crop = config.center_crop;
  const top = roundHalfEven((rh - crop) / 2);
  const left = roundHalfEven((rw - crop) / 2);
  const { mean, std, scale } = config;
  const divisor = Math.round(1 / scale); // 255: torchvision membagi, bukan mengalikan dengan 1/255
  const area = crop * crop;
  const out = new Float32Array(3 * area);
  for (let y = 0; y < crop; y++) {
    for (let x = 0; x < crop; x++) {
      const p = ((y + top) * rw + (x + left)) * 3;
      const o = y * crop + x;
      // Hitung dalam float32 seperti torchvision: ToTensor (bagi 255) lalu Normalize
      for (let c = 0; c < 3; c++) {
        const v = Math.fround(resized[p + c] / divisor);
        out[c * area + o] = Math.fround(Math.fround(v - Math.fround(mean[c])) / Math.fround(std[c]));
      }
    }
  }
  return out;
}
