# Kobutai 内容分类器

根据"标题 + 歌手或频道 + 网站 + 时长"判断正在播放的是什么，用来给 Kobutai 选角色。
完全离线，不需要 API key，不录音。一份模型文件，TypeScript / Swift / C# 三个平台共用。

## 它输出什么

| 字段 | 取值 | 用途 |
|---|---|---|
| `type` | music / video / live / podcast，或 `null` | 内容类型；`null` 表示没把握，用通用池（小喜） |
| `genre` | 10 种曲风，或 `null` | 只在 music 时有；`null` 用通用池 |
| `vclass` | ent / info，或 `null` | 只在 video 时有；ent → 爆爆，info → 听听 |

## 当前成绩（第 5 轮）

| 指标 | 目标 | 真实榜单测试集（383 条） | 合成测试集（562 条） |
|---|---|---|---|
| 内容类型准确率 | ≥ 95% | 95.6% | 97.3% |
| 曲风准确率（有把握时） | ≥ 85% | 85.5%（覆盖 93%） | 96.0%（覆盖 97%） |
| 视频娱乐/讲解准确率 | ≥ 88% | 86.2% | 95.1% |
| 模型体积 | ≤ 3 MB | 2.1 MB，另加歌手表 0.5 MB（压缩后共约 1.4 MB） | |
| 三平台结果一致 | 逐条一致 | TypeScript 559/559 条与参考实现一致；Swift、C# 已写好但还没编译运行过 | |
| 速度 | < 1 ms | Node 里约 0.94 ms 一次 | |

"真实榜单测试集"是 2026-10-13 从 Spotify、网易云、B 站、YouTube、Apple 播客等榜单页抓的真实标题，训练时没见过，是最接近实际使用的数字。
"合成测试集"由大模型凭记忆写出，和训练数据同源，分数偏高，只适合做回归检查。

### 专项：网络热梗、日本偶像、日番（360 条，各 120 条）

| | 第 4 轮 | 第 5 轮 |
|---|---|---|
| 网络热梗：完全判对 | 75.0% | 89.2% |
| 日本偶像：完全判对 | 85.0% | 90.8% |
| 日番和动漫音乐：完全判对 | 88.3% | 90.8% |

"完全判对"指类型对，并且音乐的曲风或视频的娱乐/讲解也对。这份专项集由大模型凭记忆写出（只有少数当季番名来自网页），不是真实榜单，数字偏乐观。
归类约定：meme 视频、鬼畜、音MAD → 视频/娱乐；被玩成梗的歌按歌本身的曲风算音乐；番剧正片、PV、切片 → 视频/娱乐；OP、ED、角色歌、OST → 音乐/动漫；偶像 MV 和现场 → 音乐/J-pop，偶像综艺 → 视频/娱乐，偶像直播 → 直播，偶像电台 → 播客。

### 搭配 DeepSeek（第 6 轮结论）

小模型没把握（置信度 < 0.9）时把标题交给 DeepSeek，约三分之一的内容会被转交，真实测试集准确率从 84.1% 升到 95.6%。
用 DeepSeek 的标签重新训练小模型没有效果，所以小模型保持第 5 轮。详见 `LOOP.md` 第 6 轮。

接入时注意：
- **访问凭证不能放进扩展**：任何人都能从扩展里取出来。需要一个你自己的中转服务（例如 Cloudflare Worker）保存凭证，扩展只调用中转服务，并在那里限流。
- **按"标题 + 歌手"缓存结果**：同一首歌只问一次。
- **隐私**：标题会发到服务器，需要在隐私说明里写明，默认关闭、由用户开启。
- 调用方式见 `scripts/distill_deepseek.py` 的 `ask()`，提示词就是 `spec/LABELING.md`。

## 目录

```
model/            交付物：kobutai.bin（模型）、artists.tsv（歌手表）、config.json（规则和阈值）、test-vectors.json
runtime/ts/       TypeScript 推理（可直接放进扩展），npm test 跑一致性测试
runtime/swift/    Swift 移植（未编译验证）
runtime/csharp/   C# 移植（未编译验证）
kobutai_py/       Python 参考实现，其他平台都以它为准
spec/             SPEC.md（算法规范）、LABELING.md（标签定义）
train/            训练、调阈值、评估、生成测试用例
scripts/          数据校验、歌手表合并、榜单抓取
data/raw/         训练样本、歌手名单、合成验证集（val）、真实测试集（real）、专项集（focus）
reports/          每一轮的指标和错例
PLAN.md GOAL.md LOOP.md   计划、目标、循环记录
```

## 在扩展里使用

```ts
import { Classifier } from "./classifier";
const clf = Classifier.load(config, modelArrayBuffer, artistsTsvText);
const r = clf.classify({ title, artist, source: location.hostname, duration: video.duration,
                         platformCategory: /open\.spotify\.com\/(episode|show)/.test(location.href) ? "podcast" : null });
```

`platformCategory` 是给平台自己已知的信息留的入口：Spotify 的 `/episode/` 地址、YouTube 的"音乐"分类、B 站音乐区，都应该直接传进来，比模型猜更准。

## 重新训练

```
python3 scripts/build_artists.py     # 合并歌手名单 → model/artists.tsv
python3 train/train.py               # 训练 → model/kobutai.bin（约 3 分钟）
python3 train/evaluate.py --tune --tag iterN   # 调阈值并评估 → reports/
python3 train/make_test_vectors.py   # 更新一致性测试用例
(cd runtime/ts && npm test)
```

## 已知问题和待确认

1. **Swift 和 C# 没有编译运行过。** 做桌面版时先跑各自的 `SelfTest`，对着 `model/test-vectors.json` 逐条核对。
2. **Apple 榜单数据的使用条款没有核实。** `data/apple/`、`data/raw/artists/c01_apple_charts.tsv`、`data/raw/train/t20_apple_charts.tsv` 来自 Apple 公开的榜单接口。收费上架前要么核实条款，要么去掉这三处重新训练（去掉后真实测试集曲风约降 2～3 个百分点）。
3. **Wikidata 歌手表没做成。** 查询接口对这个环境限速到每分钟 1 次，只拿到很小一部分。脚本 `scripts/fetch_wikidata.py` 可以在你的 Mac 上跑完（公共领域数据，可商用），预计能把歌手表从 2.4 万扩到 5 万以上。
4. **时长这个信号没有真正训练到。** 样本里的时长多数是估计值或缺失。
5. **真实测试集只有 637 条**（其中 383 条做最终测试），一两个百分点的差别在误差范围内。最有价值的下一步是在扩展里加一个本地开关，把真实播放过的标题存下来（不上传），攒几千条做测试集。
6. **日语乐队的 jpop / rock / anime 边界**是剩余曲风错误里最大的一类，属于标签定义本身有歧义。
