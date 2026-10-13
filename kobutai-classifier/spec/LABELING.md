# Labelling guide (for data-generation agents)

Kobutai is a browser extension / desktop app that shows a small animated
character matching whatever media is playing. It only sees **text metadata**
(title, artist or channel name, site, duration), never the audio. We are
building a tiny on-device text classifier, and you are writing its data.

Everything you write must look like what a real user would actually have
playing: real, well-known and mid-tail songs, artists, channels and shows, and
realistic titles in the exact style each site uses. Do not invent fake artists
or placeholder titles ("Song 1", "Artist A"). Titles only: never include lyrics.

## Labels

### type (what kind of content)

| value | meaning |
|---|---|
| `music` | The content is music: a track, MV, lyric video, cover, live performance, concert, playlist/mix, 24/7 music radio stream. **Music wins**: a live-streamed lofi radio is `music`, not `live`. |
| `video` | A non-music video: gaming, comedy, vlog, film/anime clip, tutorial, lecture, news, review, documentary. |
| `live` | A non-music live stream: game streaming, chatting, VTuber stream, live sports, live news. |
| `podcast` | Spoken-word audio shows: podcast episodes, radio talk shows, audiobooks, audio dramas, 有声书, 相声/评书 audio. Video recordings of podcasts on YouTube count as `podcast`. |

### genre (only when type = music, otherwise `-`)

Pick the one the *sound* of this track/artist is closest to.

| value | includes |
|---|---|
| `rock` | rock, metal, punk, alternative, indie rock, J-rock bands whose sound is rock (ONE OK ROCK, RADWIMPS → use judgment), Chinese rock bands |
| `western-pop` | English-language and other Western pop, dance-pop, R&B/soul (non-rap), Latin pop, **K-pop** |
| `jpop` | Japanese mainstream pop, idol groups, city pop, J-ballads, Japanese singer-songwriters |
| `anime` | anime OP/ED/insert songs, anisong singers, seiyuu songs, Vocaloid, utaite covers (歌ってみた), VTuber songs, game soundtracks & game songs (Genshin, Touhou arrangements), anime OSTs |
| `cpop` | Mandarin / Cantonese / other Chinese-language pop, 古风, 国风, Chinese ballads, Chinese folk-pop (民谣) |
| `electronic` | EDM, house, techno, trance, dubstep, drum & bass, synthwave, future bass, DJ sets |
| `lofi` | lofi hip hop / chillhop / study beats / chill & sleep & relax mixes, ambient, rain & café background music |
| `classical` | classical composers and performers, orchestras, piano/violin solo repertoire, opera, film scores & orchestral soundtracks, neo-classical piano |
| `hiphop` | rap, hip hop, trap, drill, in any language (Chinese rap 说唱, Japanese rap included) |
| `other` | jazz, country, folk, blues, reggae, world, children's songs, religious, musical theatre, traditional/ethnic music, anything that fits none of the above |

Tie-breaks: an anime tie-in song by a mainstream J-pop act (YOASOBI「アイドル」,
LiSA「紅蓮華」) is `anime` when the title or channel makes the anime link visible
(e.g. "TVアニメ『…』OP", an anisong singer), otherwise `jpop`. Language decides
between `cpop` / `jpop` / `western-pop` for ordinary pop. Rap in any language is
`hiphop`.

### vclass (only when type = video, otherwise `-`)

| value | meaning |
|---|---|
| `ent` | entertainment: gaming, comedy, memes, vlogs, film/TV/anime clips, variety shows, sports highlights, reactions, ASMR, mukbang, pets |
| `info` | explanatory: tutorials, courses, lectures, science & history explainers, news, interviews, tech reviews, documentaries, finance, coding, language learning |

## Other fields

- `lang`: language of the title: `zh`, `ja`, `en`, or `xx` (other / mixed).
- `domain`: the real host the user would be on, lowercase, e.g.
  `open.spotify.com`, `music.youtube.com`, `www.youtube.com`, `music.apple.com`,
  `music.163.com`, `y.qq.com`, `www.kugou.com`, `www.bilibili.com`,
  `live.bilibili.com`, `www.twitch.tv`, `soundcloud.com`, `www.nicovideo.jp`,
  `podcasts.apple.com`, `www.xiaoyuzhoufm.com`, `www.ximalaya.com`,
  `www.netflix.com`, `www.douyin.com`, `www.tiktok.com`, `www.deezer.com`,
  `tidal.com`, `bandcamp.com`, `www.iqiyi.com`, `v.qq.com`, `abema.tv`,
  `tver.jp`, `www.coursera.org`, `vimeo.com` …
- `dur`: duration in seconds as an integer. `0` = unknown (common: about 15 % of
  rows), `-1` = live / endless stream.
- `artist`: what the site reports as artist **or channel/uploader name**. On
  Spotify this is the artist; on YouTube/Bilibili it is the channel (which may
  be the artist's official channel, a label like "HYBE LABELS", a fan uploader,
  or a creator). May be empty (write nothing between the tabs) for about 5 % of rows.
- `title`: exactly as the site would show it, including brackets, 【】, emoji,
  "(Official Video)", "feat.", episode numbers, etc. No tab characters.

## File format: sample files

UTF-8 text, one sample per line, **8 tab-separated columns**, no header:

```
type<TAB>genre<TAB>vclass<TAB>lang<TAB>domain<TAB>dur<TAB>artist<TAB>title
```

Examples:

```
music	anime	-	ja	www.youtube.com	213	Ayase / YOASOBI	YOASOBI「アイドル」 Official Music Video
music	cpop	-	zh	music.163.com	269	周杰伦	晴天
music	lofi	-	en	www.youtube.com	-1	Lofi Girl	lofi hip hop radio 📚 beats to relax/study to
video	-	ent	en	www.youtube.com	1124	MrBeast	I Spent 7 Days Buried Alive
video	-	info	zh	www.bilibili.com	1502	李永乐老师	黎曼猜想到底是什么？
live	-	-	en	www.twitch.tv	-1	shroud	VALORANT ranked grind | !sponsor
podcast	-	-	en	open.spotify.com	5421	Lex Fridman Podcast	#400 – Elon Musk: War, AI, Aliens, Politics, Physics
```

## File format: artist files

UTF-8 text, one artist per line, **3 tab-separated columns**, no header:

```
name<TAB>aliases separated by ;<TAB>genre
```

- `name`: the name most commonly shown on streaming sites.
- `aliases`: other spellings users' sites may show (native script, romanised,
  English name, common abbreviation). Leave empty if none. Do not list aliases
  that are ordinary words or shorter than 2 characters.
- `genre`: one of the ten genre values above: the artist's primary sound.

```
周杰伦	Jay Chou;周杰倫	cpop
YOASOBI	ヨアソビ	jpop
LiSA	織部里沙	anime
Ludwig van Beethoven	Beethoven;贝多芬;ベートーヴェン	classical
```

## Before you finish

Run the validator on every file you wrote and fix every reported line:

```
python3 /home/claude/kobutai-classifier/scripts/validate_raw.py <file>
```

It must print `OK`. Avoid duplicates within your file.
