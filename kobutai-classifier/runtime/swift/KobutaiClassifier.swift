// Kobutai media classifier: Swift runtime (macOS / iOS).
// A line-by-line port of runtime/ts/src/classifier.ts and kobutai_py/core.py (see spec/SPEC.md).
// NOT yet compiled or run: no Swift toolchain was available where this was written.
// Before shipping, run it over model/test-vectors.json (see KobutaiSelfTest at the bottom).

import Foundation

public let kTypes = ["music", "video", "live", "podcast"]
public let kGenres = ["rock", "western-pop", "jpop", "anime", "cpop", "electronic", "lofi", "classical", "hiphop", "other"]
public let kVClasses = ["ent", "info"]

public struct MediaInput {
    public var title = ""
    public var artist = ""
    public var source = ""
    /// Seconds. nil / 0 / NaN = unknown; infinite or negative = live.
    public var duration: Double? = nil
    public var isLive = false
    /// "music" or "podcast" when the platform itself says so.
    public var platformCategory: String? = nil
    public init() {}
}

public struct Classification {
    public var type: String?
    public var typeConfidence = 0.0
    public var typeVia = "model"
    public var genre: String?
    public var genreConfidence = 0.0
    public var genreVia: String?
    public var vclass: String?
    public var vclassConfidence = 0.0
}

struct KBTConfig: Decodable {
    struct Thresholds: Decodable { let type: Double; let genre: Double; let vclass: Double }
    struct Bonus: Decodable { let artist: Double; let title: Double }
    struct Rules: Decodable { let musicSources: [String]; let podcastSources: [String]; let videoSources: [String] }
    let thresholds: Thresholds
    let artistBonus: Bonus
    let rules: Rules
    let sourceAliases: [String: String]
}

// MARK: text

func kbtCollapse(_ s: String) -> String {
    s.split(separator: " ", omittingEmptySubsequences: true).joined(separator: " ")
}

func kbtNorm(_ s: String) -> String {
    if s.isEmpty { return "" }
    var out = String.UnicodeScalarView()
    for u in s.decomposedStringWithCompatibilityMapping.unicodeScalars {
        let o = u.value
        if o >= 0x300 && o <= 0x36F { continue }
        if o >= 0x41 && o <= 0x5A { out.append(Unicode.Scalar(o + 32)!) }
        else if o == 9 || o == 10 || o == 13 || o == 11 || o == 12 || o == 0xA0 || o == 0x3000 { out.append(" ") }
        else { out.append(u) }
    }
    return kbtCollapse(String(out))
}

func kbtNormSource(_ s: String) -> String {
    var r = kbtNorm(s)
    for p in ["www.", "m."] where r.hasPrefix(p) { r = String(r.dropFirst(p.count)) }
    return r
}

func kbtDurBucket(_ d: Double?) -> String {
    guard let d = d, !d.isNaN, d != 0 else { return "U" }
    if d < 0 || d.isInfinite { return "L" }
    if d < 60 { return "a" }
    if d < 150 { return "b" }
    if d < 330 { return "c" }
    if d < 600 { return "d" }
    if d < 1500 { return "e" }
    if d < 3600 { return "f" }
    return "g"
}

func kbtFnv1a(_ s: String) -> UInt32 {
    var h: UInt32 = 0x811C9DC5
    for b in s.utf8 {
        h ^= UInt32(b)
        h = h &* 0x01000193
    }
    return h
}

func kbtWords(_ cps: [Unicode.Scalar]) -> [String] {
    var out: [String] = []
    var cur = ""
    for c in cps {
        let v = c.value
        if (v >= 0x61 && v <= 0x7A) || (v >= 0x30 && v <= 0x39) { cur.unicodeScalars.append(c) }
        else if !cur.isEmpty { out.append(cur); cur = "" }
    }
    if !cur.isEmpty { out.append(cur) }
    return out
}

func kbtFeatureStrings(title: String, artist: String, source: String, bucket: String) -> [String] {
    var feats: [String] = []
    for (tag, text) in [("t", title), ("a", artist)] {
        let cps = Array(text.unicodeScalars.prefix(160))
        var seq: [String] = ["^"]
        seq.append(contentsOf: cps.map { String($0) })
        seq.append("$")
        for n in 1...4 {
            if seq.count < n { continue }
            for i in 0...(seq.count - n) {
                let g = seq[i..<(i + n)].joined()
                feats.append("\(tag)\(n):\(g)")
                if n >= 2 { feats.append("x\(n):\(g)") }
            }
        }
        let ws = kbtWords(cps)
        for w in ws {
            feats.append("\(tag)w:\(w)")
            feats.append("xw:\(w)")
        }
        if ws.count > 1 {
            for i in 0..<(ws.count - 1) { feats.append("\(tag)b:\(ws[i]) \(ws[i + 1])") }
        }
    }
    feats.append("s:" + source)
    feats.append("d:" + bucket)
    feats.append("sd:" + source + "|" + bucket)
    return feats
}

func kbtFeatureIndices(title: String, artist: String, source: String, bucket: String, buckets: Int) -> [Int] {
    let mask = UInt32(buckets - 1)
    var set = Set<Int>()
    for f in kbtFeatureStrings(title: title, artist: artist, source: source, bucket: bucket) {
        set.insert(Int(kbtFnv1a(f) & mask))
    }
    return set.sorted()
}

// MARK: model

final class KBTModel {
    let buckets: Int
    private var heads: [(k: Int, scale: Double, bias: [Double])] = []
    private var total = 0
    private let data: Data
    private let weightsOffset: Int

    init(data: Data) throws {
        self.data = data
        func u32(_ off: Int) -> UInt32 {
            data.subdata(in: off..<(off + 4)).withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) }
        }
        func f32(_ off: Int) -> Double { Double(Float(bitPattern: u32(off))) }
        guard data.count > 16, String(data: data.subdata(in: 0..<4), encoding: .ascii) == "KBT1", u32(4) == 1 else {
            throw NSError(domain: "kobutai", code: 1, userInfo: [NSLocalizedDescriptionKey: "unsupported model file"])
        }
        buckets = Int(u32(8))
        let nheads = Int(u32(12))
        var off = 16
        for _ in 0..<nheads {
            let k = Int(u32(off))
            let scale = f32(off + 4)
            var bias: [Double] = []
            for j in 0..<k { bias.append(f32(off + 8 + 4 * j)) }
            off += 8 + 4 * k
            heads.append((k, scale, bias))
            total += k
        }
        weightsOffset = off
    }

    func logits(_ idx: [Int]) -> [[Double]] {
        var sums = [Int](repeating: 0, count: total)
        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let w = raw.baseAddress!.advanced(by: weightsOffset).assumingMemoryBound(to: Int8.self)
            for i in idx {
                let base = i * total
                for k in 0..<total { sums[k] += Int(w[base + k]) }
            }
        }
        let inv = 1.0 / Double(idx.count).squareRoot()
        var out: [[Double]] = []
        var off = 0
        for h in heads {
            var z: [Double] = []
            for j in 0..<h.k { z.append(h.bias[j] + h.scale * Double(sums[off + j]) * inv) }
            out.append(z)
            off += h.k
        }
        return out
    }
}

func kbtSoftmax(_ z: [Double], allowed: [Bool]? = nil) -> [Double] {
    var m = -Double.infinity
    for j in 0..<z.count where (allowed?[j] ?? true) && z[j] > m { m = z[j] }
    let e = z.enumerated().map { (j, v) in (allowed?[j] ?? true) ? exp(v - m) : 0.0 }
    var s = 0.0
    for v in e { s += v }
    return e.map { $0 / s }
}

func kbtArgmax(_ p: [Double]) -> Int {
    var best = 0
    for j in 1..<p.count where p[j] > p[best] { best = j }
    return best
}

// MARK: artists

let kbtSeparators: Set<Unicode.Scalar> = Set("-–—|｜/／「」『』【】()（）[]《》〈〉<>,、，&×·・:：;；\"“”'‘’~〜～!！?？#@*+=。".unicodeScalars)
let kbtCuts = [" feat", " ft ", " ft.", " x "]

func kbtTitleSegments(_ t: String) -> [String] {
    var segs: [String] = []
    var cur = String.UnicodeScalarView()
    for c in t.unicodeScalars {
        if kbtSeparators.contains(c) { segs.append(String(cur)); cur = String.UnicodeScalarView() }
        else { cur.append(c) }
    }
    segs.append(String(cur))
    var out: [String] = []
    for raw in segs {
        let s = kbtCollapse(raw)
        for cut in kbtCuts {
            if let r = s.range(of: cut), r.lowerBound > s.startIndex {
                out.append(String(s[s.startIndex..<r.lowerBound]).trimmingCharacters(in: CharacterSet(charactersIn: " ")))
            }
        }
        if !s.isEmpty { out.append(s) }
    }
    return out
}

final class KBTArtistTable {
    private var map: [String: Int] = [:]
    init(text: String) {
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            guard let tab = line.lastIndex(of: "\t"), let g = Int(line[line.index(after: tab)...]) else { continue }
            // Keys are compared by exact scalar sequence, so store them through a scalar-preserving key.
            map[kbtKey(String(line[line.startIndex..<tab]))] = g
        }
    }
    /// Swift's String == uses canonical equivalence; the reference compares code points.
    /// Keys are NFKD already, so a scalar-joined key keeps the two behaviours identical.
    private func kbtKey(_ s: String) -> String { s.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: ",") }

    func lookup(artist: String, title: String) -> (genre: Int, wherefrom: String)? {
        if !artist.isEmpty {
            if let g = map[kbtKey(artist)] { return (g, "artist") }
            for seg in kbtTitleSegments(artist) { if let g = map[kbtKey(seg)] { return (g, "artist") } }
        }
        for seg in kbtTitleSegments(title) { if let g = map[kbtKey(seg)] { return (g, "title") } }
        return nil
    }
}

// MARK: pipeline

public final class KobutaiClassifier {
    private let cfg: KBTConfig
    private let model: KBTModel
    private let artists: KBTArtistTable
    private let music: Set<String>, podcast: Set<String>, video: Set<String>

    /// configJSON: config.json, modelBin: kobutai.bin, artistsTSV: artists.tsv
    public init(configJSON: Data, modelBin: Data, artistsTSV: String) throws {
        cfg = try JSONDecoder().decode(KBTConfig.self, from: configJSON)
        model = try KBTModel(data: modelBin)
        artists = KBTArtistTable(text: artistsTSV)
        music = Set(cfg.rules.musicSources); podcast = Set(cfg.rules.podcastSources); video = Set(cfg.rules.videoSources)
    }

    public func classify(_ input: MediaInput) -> Classification {
        let t = kbtNorm(input.title), a = kbtNorm(input.artist)
        var s = kbtNormSource(input.source)
        if let alias = cfg.sourceAliases[s] { s = alias }
        let b = kbtDurBucket(input.isLive ? -1 : input.duration)
        let z = model.logits(kbtFeatureIndices(title: t, artist: a, source: s, bucket: b, buckets: model.buckets))

        var allowed: [Bool]? = nil
        var forced = -1
        if input.platformCategory == "music" || music.contains(s) { forced = 0 }
        else if input.platformCategory == "podcast" || podcast.contains(s) { forced = 3 }
        else if video.contains(s) { allowed = [false, true, false, false] }
        else if b == "L" { allowed = [true, false, true, false] }

        var res = Classification()
        var ti: Int
        if forced >= 0 {
            ti = forced; res.typeConfidence = 1; res.typeVia = "rule"
        } else {
            let p = kbtSoftmax(z[0], allowed: allowed)
            ti = kbtArgmax(p); res.typeConfidence = p[ti]
        }
        res.type = kTypes[ti]
        if res.typeConfidence < cfg.thresholds.type {
            res.type = nil
        } else if ti == 0 {
            var zg = z[1]
            let hit = artists.lookup(artist: a, title: t)
            if let hit = hit { zg[hit.genre] += hit.wherefrom == "artist" ? cfg.artistBonus.artist : cfg.artistBonus.title }
            let p = kbtSoftmax(zg)
            let gi = kbtArgmax(p)
            res.genreConfidence = p[gi]
            if p[gi] >= cfg.thresholds.genre {
                res.genre = kGenres[gi]
                res.genreVia = (hit != nil && hit!.genre == gi) ? "table" : "model"
            }
        } else if ti == 1 {
            let p = kbtSoftmax(z[2])
            let vi = kbtArgmax(p)
            res.vclassConfidence = p[vi]
            if p[vi] >= cfg.thresholds.vclass { res.vclass = kVClasses[vi] }
        }
        return res
    }
}

// MARK: self-test against model/test-vectors.json

public enum KobutaiSelfTest {
    /// Returns the number of mismatching cases (0 = this port agrees with the reference).
    public static func run(classifier: KobutaiClassifier, vectorsJSON: Data) throws -> Int {
        guard let root = try JSONSerialization.jsonObject(with: vectorsJSON) as? [String: Any],
              let cases = root["cases"] as? [[String: Any]] else { return -1 }
        var bad = 0
        for c in cases {
            guard let i = c["input"] as? [String: Any], let e = c["expected"] as? [String: Any] else { continue }
            var m = MediaInput()
            m.title = i["title"] as? String ?? ""
            m.artist = i["artist"] as? String ?? ""
            m.source = i["source"] as? String ?? ""
            m.duration = (i["duration"] as? NSNumber)?.doubleValue
            m.isLive = i["isLive"] as? Bool ?? false
            m.platformCategory = i["platformCategory"] as? String
            let r = classifier.classify(m)
            func close(_ x: Double, _ key: String) -> Bool { abs(x - ((e[key] as? NSNumber)?.doubleValue ?? 0)) < 1e-9 }
            let same = r.type == (e["type"] as? String) && r.genre == (e["genre"] as? String)
                && r.vclass == (e["vclass"] as? String) && r.genreVia == (e["genreVia"] as? String)
                && close(r.typeConfidence, "typeConfidence") && close(r.genreConfidence, "genreConfidence")
                && close(r.vclassConfidence, "vclassConfidence")
            if !same { bad += 1 }
        }
        return bad
    }
}
