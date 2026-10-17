# 可运行Demo与视频材料

Demo通过SGLang的OpenAI兼容流接口展示文本问答。已归档 [真实API终端回放视频](demo/demo.mp4)，时长11.854秒、125,545字节，SHA256为 `f8e6d9b4dc94e5fcb2a8ea9ad08c493da60c5fe067c3b645069d7b8516602ad6`；已完整解码核验。它根据真实SSE事件时间流生成，属于终端回放，不能当作GUI录屏或精度成绩。

## 启动和访问

在服务器按 [README第1–3节](../README.md) 完成环境、物理卡0、1容器创建、依赖检查和模型启动，随后执行：

```bash
bash scripts/run_review01.sh demo '请用中文简短解释张量并行。'
```

API默认 `http://127.0.0.1:31014/v1/chat/completions`，模型名 `Qwen3.8-27B-Channel-INT8-w8a8`。服务host=0.0.0.0，可通过PORT配置；外部访问使用服务器可达地址。也可在宿主直接请求：

```bash
curl --noproxy '*' -N http://127.0.0.1:31014/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-Channel-INT8-w8a8","messages":[{"role":"user","content":"请简要介绍张量并行。"}],"max_tokens":256,"temperature":0,"stream":true}'
```

演示后执行 `bash scripts/run_review01.sh stop`，只停止已验证专属身份的评审容器，释放卡0、1的协作锁。历史 `scripts/demo.sh` 与 `scripts/run.sh` 是物理6、7入口；本页使用与当前README一致的0、1入口。

## 原始演示证据

原始请求、SSE事件时间和SHA见 [capture.json](../result/evidence/official-alignment-20261017/community-official-followup-20261017-d/runtime-p23/demo/capture.json)。视频用于展示真实访问方式；性能数据以正式Benchmark原件为准，图文和模型精度由对应测试验证。
