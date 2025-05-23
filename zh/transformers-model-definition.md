---
title: "Transformers 库：进行模型定义的标准化工作 (The Transformers Library: standardizing model definitions)"
thumbnail: /blog/assets/transformers-model-definition/transformers-thumbnail.png
authors:
- user: lysandre
- user: ArthurZ
- user: pcuenq
- user: julien-c
translators:
- user: smartisan
---

# Transformers 库：进行模型定义的标准化工作 

内容简介：展望未来，我们的目标是把Transformers这个库打造成一个能够适配不同框架的核心纽带：换句话说，如果某一个模型架构适配了Transformers库，那么大家就可以期待，这个架构同样也会在生态系统的其余组成部分当中获得相应的支持。

---

Transformers 这个库是在 2019 年被创建出来的，紧随在 BERT Transformer 模型发布之后。从那时以来，我们一直致力于去增添最为先进的那些架构，起初的工作重心主要放在自然语言处理，也就是 NLP 这个领域，后续则逐步扩展到了音频 以及计算机视觉这两个方向。发展到如今，Transformers 库已经成为了 Python 这个生态系统当中，针对大语言模型 (LLM) 以及视觉语言模型 (VLM) 的一个默认选项库。

就目前来看，Transformers 库能够支持到超过 300 种不同的模型架构，并且平均算下来，每个星期都会有大约 3 种新的架构被添加进来。我们相关工作的目标一直都是要及时地去适配这些架构；特别是那些最受大家追捧的架构，比如说像 Llamas、Qwens、以及 GLMs 等等，去提供一种最及时支持，也就是 day-0 support。

## 一个模型定义的库

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers-model-definition/transformers-thumbnail.png" alt="Transformers 模型定义的标准化">
随着时间的不断发展，Transformers 库已经演变成为了机器学习生态系统当中的一个核心组成部分，也就是 中心化组件，并且在模型多样性这个方面，它也成为了最为完备的工具包集合当中的一员；它已经被集成到了所有主流的那些训练框架，即 training frameworks 里面，比如说像 Axolotl、Unsloth、DeepSpeed、FSDP、PyTorch-Lightning、TRL、以及 Nanotron 等等这些框架。

最近，我们的相关工作一直在和那些最为流行的推理引擎，比如像 vLLM、SGLang、以及 TGI 等，去进行非常紧密的合作，目的在于推动它们把 `transformers` 当作后端，也就是 backend 来加以运用。这样做所能够带来的价值可以说是非常显著的：一旦有某一个模型被添加到了 `transformers` 当中，那么这个模型就能够在这些推理引擎里面得到运用，*并且在这个过程当中，还能够充分借助每个引擎自身所具备的那些强项*：例如推理方面的优化，也就是 inference optimizations、专门定制的内核，即 specialized kernels、以及动态批处理技术，也就是 dynamic batching 等等这些方面。

作为一个例子，下面就来展示一下在 vLLM 这个工具里面，要如何去运用 `transformers` 这个后端：

```python
from vllm import LLM

llm = LLM(model="new-transformers-model", model_impl="transformers")
```

只需要这么简单的几步，一个新的模型就能够借助 vLLM 来享受到速度超快并且达到生产级别的服务体验了！

想要了解更多这方面的相关信息，可以去查阅 [vLLM 的相关文档](https://blog.vllm.ai/2025/04/11/transformers-backend.html)。

---

此外，相关工作还和 [llama.cpp](https://github.com/ggml-org/llama.cpp) 以及 [MLX](https://github.com/ml-explore/mlx) 这两个项目进行了程度非常密切的合作，其目的在于确保 `transformers` 和这些建模库之间的具体实现，能够拥有一种良好的互操作性，也就是 interoperability。举个例子来说，正是得益于社区方面所付出的大量努力，现在已经可以非常轻松地在 `transformers` 当中去[加载 GGUF 文件](https://huggingface.co/docs/transformers/en/gguf)，从而可以去进行后续的微调工作，也就是 fine-tuning。反过来看，Transformers 的模型也同样可以很方便地被[转换成为 GGUF 文件](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)，以便能够给 llama.cpp 来进行使用。

对于 MLX 这个项目来说，情况也是类似的，Transformers 所产生的 safetensors 文件，是能够直接和 MLX 的那些模型实现兼容的。对于 `transformers` 这种格式能够被社区所广泛采纳这件事情，相关方面感到非常自豪，因为这样做带来了极大的互操作性，使得所有人都能够从中获益。大家可以运用 Unsloth 来进行模型的训练工作，再借助 SGLang 去进行部署，随后还可以把这个模型导出到 llama.cpp 当中，在本地环境去运行它！后续的目标是会继续为社区的向前发展去提供支持。

## 努力去实现更为简化的模型贡献流程

为了能够让广大的社区成员更加容易地把 Transformers 这个库当作模型定义的一个参考标准来加以运用，相关工作一直致力于去极大程度上降低模型贡献，也就是 模型贡献度 的门槛。这方面的工作实际上已经开展了好几年了，不过在接下来的几个星期里面，这个进程将会得到显著的加速：
- 每一款模型的建模相关代码，都将会得到进一步的简化处理；并且会为那些最为关键的核心组件，比如说像 KV 缓存 (KV cache)、各种不同的注意力函数 (Attention functions)、以及内核优化 (kernel optimization) 等，去提供一套清晰而且简洁的 API 接口。
- 相关计划包括将会去弃用那些冗余的组件，也就是 deprecate redundant components，从而转向运用一种简单并且统一的方式来对这些 API 进行调用：具体做法包括会通过去弃用那些处理速度较慢的分词器，即 slow tokenizers，来鼓励大家选用高效的分词方法，也就是 tokenization；并且，与之类似地，也会推荐去选用那些快速的向量化视觉处理器，即 fast vectorized vision processors。
- 还会继续去加强那些围绕着*模块化*模型定义，也就是 modular model definitions 所开展的工作，其目标在于要让新的模型在添加时，只需要进行绝对最小限度的代码修改工作。像过去那种动辄需要贡献 6000 行代码、涉及 20 个文件变更的新模型提交方式，都将成为过去式了。

## 这对您具体会带来怎样的影响？

### 这对您作为一名模型用户，具体会带来哪些影响 

作为一名模型的使用者，在未来应该能够观察到，您所运用的各种工具之间将会展现出更强的操作关联性。这一点并非是要想让大家必须要在实验过程当中去运用 `transformers` 这个库；恰恰相反，它所要表达的意思是，正是得益于这种模型定义方面的标准化工作，大家可以合理地使用那些被您用来进行训练、执行推理、以及投入生产环境的各类工具，并且一起高效地开展协同工作。

### 这对您作为一名模型创建者，具体会带来哪些影响 

作为一名模型的创建者，这一点意味着，仅仅通过一次贡献，就能够让您所创建的模型，在所有那些已经集成了这种建模实现的下游库当中，都能够被投入使用。在过去的很多年当中，我们已经多次观察到这样的情况：去进行一个模型的发布工作，往往伴随着巨大的压力，并且，要把这个模型集成到所有重要的库里面去，通常都会演变成一项非常消耗时间的任务。通过运用一种社区驱动的方式，也就是 community-driven manner，来对模型的实现进行标准化的工作，我们期望能够借此来降低在各个库之间对这个领域去进行贡献的门槛。

---

我们抱有坚定的信念，认为这个全新的发展方向，将会有助于去对一个时常面临碎片化，也就是 fragmentation 风险的生态系统，进行标准化的工作。我们也非常期望能够聆听到大家对于团队目前所选定的这个发展方向的反馈意见；同时也包括我们可以去做出哪些调整以来更好地达成这个目标。欢迎大家前往 Hub 平台上的 [transformers-community 支持板块](https://huggingface.co/spaces/transformers-community/support)，来和我们进行交流！

