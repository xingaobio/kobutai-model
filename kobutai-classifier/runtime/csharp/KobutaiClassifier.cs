// Kobutai media classifier: C# runtime (Windows, .NET 6+).
// A line-by-line port of runtime/ts/src/classifier.ts and kobutai_py/core.py (see spec/SPEC.md).
// NOT yet compiled or run: no .NET toolchain was available where this was written.
// Before shipping, run it over model/test-vectors.json (see SelfTest at the bottom).

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Kobutai
{
    public sealed class MediaInput
    {
        public string Title = "";
        public string Artist = "";
        public string Source = "";
        /// <summary>Seconds. null / 0 / NaN = unknown; infinite or negative = live.</summary>
        public double? Duration;
        public bool IsLive;
        /// <summary>"music" or "podcast" when the platform itself says so.</summary>
        public string? PlatformCategory;
    }

    public sealed class Classification
    {
        public string? Type;
        public double TypeConfidence;
        public string TypeVia = "model";
        public string? Genre;
        public double GenreConfidence;
        public string? GenreVia;
        public string? VClass;
        public double VClassConfidence;
    }

    public sealed class Classifier
    {
        public static readonly string[] Types = { "music", "video", "live", "podcast" };
        public static readonly string[] Genres = { "rock", "western-pop", "jpop", "anime", "cpop", "electronic", "lofi", "classical", "hiphop", "other" };
        public static readonly string[] VClasses = { "ent", "info" };

        // ---- config
        readonly double thType, thGenre, thVClass, bonusArtist, bonusTitle;
        readonly HashSet<string> music, podcast, video;
        readonly Dictionary<string, string> aliases = new(StringComparer.Ordinal);

        // ---- model
        readonly int buckets, total;
        readonly List<(int k, double scale, double[] bias)> heads = new();
        readonly sbyte[] w;

        // ---- artists (ordinal comparison: the reference compares code points, never culture rules)
        readonly Dictionary<string, int> artists = new(StringComparer.Ordinal);

        public Classifier(string configJson, byte[] modelBin, string artistsTsv)
        {
            using var doc = JsonDocument.Parse(configJson);
            var root = doc.RootElement;
            var th = root.GetProperty("thresholds");
            thType = th.GetProperty("type").GetDouble();
            thGenre = th.GetProperty("genre").GetDouble();
            thVClass = th.GetProperty("vclass").GetDouble();
            bonusArtist = root.GetProperty("artistBonus").GetProperty("artist").GetDouble();
            bonusTitle = root.GetProperty("artistBonus").GetProperty("title").GetDouble();
            var rules = root.GetProperty("rules");
            HashSet<string> Read(string name) =>
                new(rules.GetProperty(name).EnumerateArray().Select(x => x.GetString()!), StringComparer.Ordinal);
            music = Read("musicSources"); podcast = Read("podcastSources"); video = Read("videoSources");
            foreach (var p in root.GetProperty("sourceAliases").EnumerateObject()) aliases[p.Name] = p.Value.GetString()!;

            if (modelBin.Length < 16 || Encoding.ASCII.GetString(modelBin, 0, 4) != "KBT1" || BitConverter.ToUInt32(modelBin, 4) != 1)
                throw new InvalidDataException("kobutai: unsupported model file");
            buckets = (int)BitConverter.ToUInt32(modelBin, 8);          // file is little-endian, as is every Windows target
            int nheads = (int)BitConverter.ToUInt32(modelBin, 12);
            int off = 16;
            for (int h = 0; h < nheads; h++)
            {
                int k = (int)BitConverter.ToUInt32(modelBin, off);
                double scale = BitConverter.ToSingle(modelBin, off + 4);
                var bias = new double[k];
                for (int j = 0; j < k; j++) bias[j] = BitConverter.ToSingle(modelBin, off + 8 + 4 * j);
                off += 8 + 4 * k;
                heads.Add((k, scale, bias));
                total += k;
            }
            w = new sbyte[buckets * total];
            Buffer.BlockCopy(modelBin, off, w, 0, w.Length);

            foreach (var line in artistsTsv.Split('\n'))
            {
                int tab = line.LastIndexOf('\t');
                if (tab > 0 && int.TryParse(line.AsSpan(tab + 1), out int g)) artists[line.Substring(0, tab)] = g;
            }
        }

        // ---------------------------------------------------------------- text
        static string Collapse(string s) => string.Join(" ", s.Split(' ', StringSplitOptions.RemoveEmptyEntries));

        public static string Norm(string? s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            var sb = new StringBuilder();
            foreach (var r in s.Normalize(NormalizationForm.FormKD).EnumerateRunes())
            {
                int o = r.Value;
                if (o >= 0x300 && o <= 0x36F) continue;
                if (o >= 0x41 && o <= 0x5A) sb.Append((char)(o + 32));
                else if (o == 9 || o == 10 || o == 13 || o == 11 || o == 12 || o == 0xA0 || o == 0x3000) sb.Append(' ');
                else sb.Append(r.ToString());
            }
            return Collapse(sb.ToString());
        }

        public static string NormSource(string? s)
        {
            var r = Norm(s);
            foreach (var p in new[] { "www.", "m." })
                if (r.StartsWith(p, StringComparison.Ordinal)) r = r.Substring(p.Length);
            return r;
        }

        public static string DurBucket(double? d)
        {
            if (d is null || double.IsNaN(d.Value) || d.Value == 0) return "U";
            double v = d.Value;
            if (v < 0 || double.IsInfinity(v)) return "L";
            if (v < 60) return "a";
            if (v < 150) return "b";
            if (v < 330) return "c";
            if (v < 600) return "d";
            if (v < 1500) return "e";
            if (v < 3600) return "f";
            return "g";
        }

        public static uint Fnv1a(string s)
        {
            uint h = 0x811C9DC5;
            foreach (byte b in Encoding.UTF8.GetBytes(s))
            {
                h ^= b;
                unchecked { h *= 0x01000193; }
            }
            return h;
        }

        static List<string> Words(List<string> cps)
        {
            var o = new List<string>();
            var cur = new StringBuilder();
            foreach (var c in cps)
            {
                bool wc = c.Length == 1 && ((c[0] >= 'a' && c[0] <= 'z') || (c[0] >= '0' && c[0] <= '9'));
                if (wc) cur.Append(c);
                else if (cur.Length > 0) { o.Add(cur.ToString()); cur.Clear(); }
            }
            if (cur.Length > 0) o.Add(cur.ToString());
            return o;
        }

        public static List<string> FeatureStrings(string title, string artist, string source, string bucket)
        {
            var feats = new List<string>();
            foreach (var (tag, text) in new[] { ("t", title), ("a", artist) })
            {
                var cps = text.EnumerateRunes().Take(160).Select(r => r.ToString()).ToList();
                var seq = new List<string> { "^" };
                seq.AddRange(cps);
                seq.Add("$");
                for (int n = 1; n <= 4; n++)
                    for (int i = 0; i + n <= seq.Count; i++)
                    {
                        string g = string.Concat(seq.Skip(i).Take(n));
                        feats.Add(tag + n + ":" + g);
                        if (n >= 2) feats.Add("x" + n + ":" + g);
                    }
                var ws = Words(cps);
                foreach (var word in ws) { feats.Add(tag + "w:" + word); feats.Add("xw:" + word); }
                for (int i = 0; i + 1 < ws.Count; i++) feats.Add(tag + "b:" + ws[i] + " " + ws[i + 1]);
            }
            feats.Add("s:" + source);
            feats.Add("d:" + bucket);
            feats.Add("sd:" + source + "|" + bucket);
            return feats;
        }

        public static int[] FeatureIndices(string title, string artist, string source, string bucket, int buckets)
        {
            uint mask = (uint)(buckets - 1);
            var set = new HashSet<int>();
            foreach (var f in FeatureStrings(title, artist, source, bucket)) set.Add((int)(Fnv1a(f) & mask));
            var arr = set.ToArray();
            Array.Sort(arr);
            return arr;
        }

        // ---------------------------------------------------------------- model
        double[][] Logits(int[] idx)
        {
            var sums = new long[total];
            foreach (int i in idx)
            {
                int b = i * total;
                for (int k = 0; k < total; k++) sums[k] += w[b + k];
            }
            double inv = 1.0 / Math.Sqrt(idx.Length);
            var o = new double[heads.Count][];
            int off = 0;
            for (int h = 0; h < heads.Count; h++)
            {
                var (k, scale, bias) = heads[h];
                var z = new double[k];
                for (int j = 0; j < k; j++) z[j] = bias[j] + scale * sums[off + j] * inv;
                o[h] = z;
                off += k;
            }
            return o;
        }

        static double[] Softmax(double[] z, bool[]? allowed = null)
        {
            double m = double.NegativeInfinity;
            for (int j = 0; j < z.Length; j++) if ((allowed == null || allowed[j]) && z[j] > m) m = z[j];
            var e = new double[z.Length];
            for (int j = 0; j < z.Length; j++) e[j] = (allowed == null || allowed[j]) ? Math.Exp(z[j] - m) : 0.0;
            double s = 0;
            foreach (double v in e) s += v;
            for (int j = 0; j < e.Length; j++) e[j] /= s;
            return e;
        }

        static int Argmax(double[] p)
        {
            int best = 0;
            for (int j = 1; j < p.Length; j++) if (p[j] > p[best]) best = j;
            return best;
        }

        // ---------------------------------------------------------------- artists
        const string Separators = "-–—|｜/／「」『』【】()（）[]《》〈〉<>,、，&×·・:：;；\"“”'‘’~〜～!！?？#@*+=。";
        static readonly string[] Cuts = { " feat", " ft ", " ft.", " x " };

        static List<string> TitleSegments(string t)
        {
            var segs = new List<string>();
            var cur = new StringBuilder();
            foreach (char c in t)                 // every separator is in the BMP, so chars are safe here
            {
                if (Separators.IndexOf(c) >= 0) { segs.Add(cur.ToString()); cur.Clear(); }
                else cur.Append(c);
            }
            segs.Add(cur.ToString());
            var o = new List<string>();
            foreach (var raw in segs)
            {
                var s = Collapse(raw);
                foreach (var cut in Cuts)
                {
                    int p = s.IndexOf(cut, StringComparison.Ordinal);
                    if (p > 0) o.Add(s.Substring(0, p).Trim(' '));
                }
                if (s.Length > 0) o.Add(s);
            }
            return o;
        }

        (int genre, string where)? Lookup(string artist, string title)
        {
            if (artist.Length > 0)
            {
                if (artists.TryGetValue(artist, out int g)) return (g, "artist");
                foreach (var seg in TitleSegments(artist)) if (artists.TryGetValue(seg, out g)) return (g, "artist");
            }
            foreach (var seg in TitleSegments(title)) if (artists.TryGetValue(seg, out int g2)) return (g2, "title");
            return null;
        }

        // ---------------------------------------------------------------- pipeline
        public Classification Classify(MediaInput input)
        {
            string t = Norm(input.Title), a = Norm(input.Artist);
            string s = NormSource(input.Source);
            if (aliases.TryGetValue(s, out var alias)) s = alias;
            string b = DurBucket(input.IsLive ? -1 : input.Duration);
            var z = Logits(FeatureIndices(t, a, s, b, buckets));

            bool[]? allowed = null;
            int forced = -1;
            if (input.PlatformCategory == "music" || music.Contains(s)) forced = 0;
            else if (input.PlatformCategory == "podcast" || podcast.Contains(s)) forced = 3;
            else if (video.Contains(s)) allowed = new[] { false, true, false, false };
            else if (b == "L") allowed = new[] { true, false, true, false };

            var res = new Classification();
            int ti;
            if (forced >= 0) { ti = forced; res.TypeConfidence = 1; res.TypeVia = "rule"; }
            else
            {
                var p = Softmax(z[0], allowed);
                ti = Argmax(p);
                res.TypeConfidence = p[ti];
            }
            res.Type = Types[ti];
            if (res.TypeConfidence < thType) res.Type = null;
            else if (ti == 0)
            {
                var zg = (double[])z[1].Clone();
                var hit = Lookup(a, t);
                if (hit != null) zg[hit.Value.genre] += hit.Value.where == "artist" ? bonusArtist : bonusTitle;
                var p = Softmax(zg);
                int gi = Argmax(p);
                res.GenreConfidence = p[gi];
                if (p[gi] >= thGenre)
                {
                    res.Genre = Genres[gi];
                    res.GenreVia = (hit != null && hit.Value.genre == gi) ? "table" : "model";
                }
            }
            else if (ti == 1)
            {
                var p = Softmax(z[2]);
                int vi = Argmax(p);
                res.VClassConfidence = p[vi];
                if (p[vi] >= thVClass) res.VClass = VClasses[vi];
            }
            return res;
        }

        // ---------------------------------------------------------------- self-test
        /// <summary>Runs model/test-vectors.json. Returns the number of mismatching cases (0 = agrees with the reference).</summary>
        public int SelfTest(string vectorsJson)
        {
            using var doc = JsonDocument.Parse(vectorsJson);
            int bad = 0;
            foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
            {
                var i = c.GetProperty("input");
                var e = c.GetProperty("expected");
                var m = new MediaInput
                {
                    Title = i.GetProperty("title").GetString() ?? "",
                    Artist = i.GetProperty("artist").GetString() ?? "",
                    Source = i.GetProperty("source").GetString() ?? "",
                    Duration = i.GetProperty("duration").ValueKind == JsonValueKind.Number ? i.GetProperty("duration").GetDouble() : null,
                    IsLive = i.GetProperty("isLive").GetBoolean(),
                    PlatformCategory = i.GetProperty("platformCategory").GetString(),
                };
                var r = Classify(m);
                static string? S(JsonElement x) => x.ValueKind == JsonValueKind.String ? x.GetString() : null;
                bool same = r.Type == S(e.GetProperty("type")) && r.Genre == S(e.GetProperty("genre"))
                    && r.VClass == S(e.GetProperty("vclass")) && r.GenreVia == S(e.GetProperty("genreVia"))
                    && Math.Abs(r.TypeConfidence - e.GetProperty("typeConfidence").GetDouble()) < 1e-9
                    && Math.Abs(r.GenreConfidence - e.GetProperty("genreConfidence").GetDouble()) < 1e-9
                    && Math.Abs(r.VClassConfidence - e.GetProperty("vclassConfidence").GetDouble()) < 1e-9;
                if (!same) bad++;
            }
            return bad;
        }
    }
}
