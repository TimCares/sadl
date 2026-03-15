# CHANGELOG

<!-- version list -->

## v1.4.0 (2026-03-15)

### Bug Fixes

- **makefile**: Adjusted uv dev install cmd
  ([`df97b14`](https://github.com/TimCares/sadl/commit/df97b14ec0b016cc16d1024da0c421021b775e8d))

- **parameter**: Fixed error where Parameter requires_grad was set to False upon creation when
  global grad tracking was disabled
  ([`a5bb761`](https://github.com/TimCares/sadl/commit/a5bb761a0a44750c6b158dfed8a2204c2d4bf5ff))

- **serialization**: Adjusted saving and loading strategy of Tensors and Function state
  ([`e352d04`](https://github.com/TimCares/sadl/commit/e352d042c3f805327abfed42c250e27fdfab8b42))

### Chores

- Update badges
  ([`a286d95`](https://github.com/TimCares/sadl/commit/a286d95beb0f9aa175d161d59561ba0247f1a5dd))

### Documentation

- Added first structure and md files for documentation
  ([`0271df9`](https://github.com/TimCares/sadl/commit/0271df9548785f35bd6468c4ac75706f35b61a17))

- Improved doc structure and added content about the motivation
  ([`9c60e7e`](https://github.com/TimCares/sadl/commit/9c60e7e18b26f29a5c77dfd1f5a1918b5de175d3))

- **autograd**: Added docs on backpropagation
  ([`151b17b`](https://github.com/TimCares/sadl/commit/151b17b5176c1657e66b37846d29d9b29efb3ef9))

- **backend**: Added docs on the backend design philosophy
  ([`1bd0d60`](https://github.com/TimCares/sadl/commit/1bd0d60b17da2822c7aa741503d2ed62e12b58c8))

- **contributing**: Added note on uv
  ([`0540349`](https://github.com/TimCares/sadl/commit/054034911110763b2cb7289658bb5f974c8e2d65))

- **getting-started**: Adjusted phrasing
  ([`b478ca2`](https://github.com/TimCares/sadl/commit/b478ca296c0b46a0de7c55e5ed9552f805f65b99))

- **notebook**: Updated demo
  ([`e22e5d4`](https://github.com/TimCares/sadl/commit/e22e5d489bcaf9997fb6c10a423ca3b62ac8dc27))

- **README**: Slight restructuring of content
  ([`18461c3`](https://github.com/TimCares/sadl/commit/18461c31f6e0765c003c1eaa3512be42e9c637f3))

- **Tensor**: Added docs on Tensor class
  ([`c94cedd`](https://github.com/TimCares/sadl/commit/c94cedd0fd42cb6a3a0c8ab0194129be352b9ec0))

### Features

- **backend**: Completely refactored backend strategy based on cupy/numpy interoperability
  ([`6061cf1`](https://github.com/TimCares/sadl/commit/6061cf1952dcdf680a685b223eed0ff0cdd2faeb))

- **gpu**: Complete redesign of the framework to support cpu+gpu simultaneously
  ([`e3b41f4`](https://github.com/TimCares/sadl/commit/e3b41f42ffdd9b4900d62e8e9514786971ca48e8))

### Refactoring

- **misc**: Restructured files to prepare for cupy/numpy refactoring
  ([`0d9ef16`](https://github.com/TimCares/sadl/commit/0d9ef167a4a907f2add903dedf91a73ab6fab762))

- **ops**: Removed obsolete import
  ([`79960c1`](https://github.com/TimCares/sadl/commit/79960c12ee046020a3018f146736d768a6f164aa))


## v1.3.0 (2026-02-11)

### Features

- **optimizer**: Added AdamW, SGD w/ momentum, and SGDW
  ([`a8dde89`](https://github.com/TimCares/sadl/commit/a8dde89641313296577ce8df7d4641de914397f1))


## v1.2.2 (2026-02-08)

### Bug Fixes

- **device**: Fixed device handling for cupy
  ([`b6cb9d6`](https://github.com/TimCares/sadl/commit/b6cb9d657ed0b1c175a922284de5661674d9c876))


## v1.2.1 (2026-02-08)

### Bug Fixes

- **optimizer**: Removed params attribute from optimizer state
  ([`34f4a16`](https://github.com/TimCares/sadl/commit/34f4a164809e22c6a25ad110de35b1d00ea7a6fe))


## v1.2.0 (2026-02-07)

### Documentation

- **readme**: Added link to demo
  ([`eff3634`](https://github.com/TimCares/sadl/commit/eff3634a96cddfb3857ce9a263d16c3e368205d7))

### Features

- **optimizer**: Added Adam optimizer
  ([`f7ad9be`](https://github.com/TimCares/sadl/commit/f7ad9be656f1caa80b27a94071d91040a92bba09))


## v1.1.0 (2026-02-06)

### Documentation

- **notebook**: Fixed mnist demo
  ([`17f4c32`](https://github.com/TimCares/sadl/commit/17f4c321b8de71f6faee20f46cf666e896a4dc36))

### Features

- **function**: Added Softmax and LogSoftmax
  ([`18b0255`](https://github.com/TimCares/sadl/commit/18b02553754e89c7f34dd0cd32a2df89a6237979))


## v1.0.2 (2026-02-05)

### Bug Fixes

- **ci**: Reordered build strategy
  ([`4969506`](https://github.com/TimCares/sadl/commit/49695067dec41bd9faff2872a31bf845e6707055))


## v1.0.1 (2026-02-05)

### Bug Fixes

- **ci**: Added github release to publish step
  ([`def70eb`](https://github.com/TimCares/sadl/commit/def70eb4f329f36f53ffa0cdc3713967eca91c99))

### Documentation

- **grad-op**: Adjusted documentation for new register_grad_op decorator
  ([`3f450aa`](https://github.com/TimCares/sadl/commit/3f450aa612083d61f23e1888ceeed529694ba17d))


## v1.0.0 (2026-02-04)

- Initial Release
