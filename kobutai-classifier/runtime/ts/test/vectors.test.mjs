// Runs every case in model/test-vectors.json through the TypeScript runtime and compares with the Python reference.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { Classifier, featureIndices, norm, normSource, durBucket } from "../dist/classifier.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "../../../model");
const bin = readFileSync(join(root, "kobutai.bin"));
const clf = Classifier.load(JSON.parse(readFileSync(join(root, "config.json"), "utf8")),
  bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength), readFileSync(join(root, "artists.tsv"), "utf8"));
const vectors = JSON.parse(readFileSync(join(root, "test-vectors.json"), "utf8"));
let bad = 0;
const t0 = performance.now();
for (const v of vectors.cases) {
  const i = v.input;
  const r = clf.classify({ title: i.title, artist: i.artist, source: i.source, duration: i.duration,
    isLive: i.isLive, platformCategory: i.platformCategory });
  const e = v.expected;
  const close = (x, y) => Math.abs(x - y) < 1e-9;
  const same = r.type === e.type && r.genre === e.genre && r.vclass === e.vclass && r.typeVia === e.typeVia &&
    r.genreVia === e.genreVia && close(r.typeConfidence, e.typeConfidence) &&
    close(r.genreConfidence, e.genreConfidence) && close(r.vclassConfidence, e.vclassConfidence);
  let featOk = true;
  if (v.featureHash !== undefined) {
    const idx = featureIndices(norm(i.title), norm(i.artist), normSource(i.source), durBucket(i.isLive ? -1 : i.duration), vectors.buckets);
    let h = 0;
    for (const x of idx) h = (Math.imul(h, 31) + x) >>> 0;
    featOk = h === v.featureHash;
  }
  if (!same || !featOk) {
    bad++;
    if (bad <= 5) console.log("MISMATCH", JSON.stringify(i), "\n  got     ", JSON.stringify(r), "\n  expected", JSON.stringify(e), featOk ? "" : "(feature indices differ)");
  }
}
const ms = (performance.now() - t0) / vectors.cases.length;
console.log(`${vectors.cases.length - bad}/${vectors.cases.length} cases match the Python reference; ${ms.toFixed(3)} ms per classification`);
process.exit(bad ? 1 : 0);
