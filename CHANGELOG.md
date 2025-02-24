# Changelog

## [0.2.0](https://github.com/tpnam0901/QTSeg/compare/v0.1.0...v0.2.0) (2025-02-24)


### Features

* add acdc, fives fold split ([b7cf29a](https://github.com/tpnam0901/QTSeg/commit/b7cf29a554a38ef12811fbfd7908ffda48d54bae))
* add dsb, kvasir preproccess ([bd345bf](https://github.com/tpnam0901/QTSeg/commit/bd345bfbf83723e8f2dc55ac6a1079871e6df355))
* add DSB2018, KvasirSEG dataset ([863abea](https://github.com/tpnam0901/QTSeg/commit/863abea32e91dee784455da4affa768b5db485d2))
* add evaluation at original size ([917a5ae](https://github.com/tpnam0901/QTSeg/commit/917a5ae965a0cddb5a6c7230cabbf09965633e56))
* add FIVES dataset, fix batch size for evaluation ([819f38f](https://github.com/tpnam0901/QTSeg/commit/819f38f3524b5c8a4996e455f67667e541fbf781))
* add PolyLR schedulers ([d9e9eb0](https://github.com/tpnam0901/QTSeg/commit/d9e9eb097ebd9e689b4f77889ea06c41a533c0eb))
* add pretrained evaluation to colab link ([d00df0b](https://github.com/tpnam0901/QTSeg/commit/d00df0b9d942aa234f675c7278b74c4ee78eee3e))
* add test dataset ([b44077a](https://github.com/tpnam0901/QTSeg/commit/b44077a058c9eb4b0d85ebeec91ad201537919d2))
* apply min-max normalization ([e662699](https://github.com/tpnam0901/QTSeg/commit/e66269969e6f23281ff79b0b9425e3af98baf23e))
* change preprocessing image ([3890228](https://github.com/tpnam0901/QTSeg/commit/38902288c894e817bfec5bf60b41e84408bd1332))
* **configs:** change defaul config ([4a6c3b4](https://github.com/tpnam0901/QTSeg/commit/4a6c3b4f687051efee136785eed81d469d3cbbad))
* **eval:** add evaluation at resize ([41fd7cb](https://github.com/tpnam0901/QTSeg/commit/41fd7cb3d534d9036fa67889e8404c9b31483ef1))
* **SEED:** add dynamic seed ([bd92abd](https://github.com/tpnam0901/QTSeg/commit/bd92abd33dcbcbf94059a2b901c13796392f7d0c))


### Bug Fixes

* **dataset:** augmentation dataset ([da0f43b](https://github.com/tpnam0901/QTSeg/commit/da0f43b75859adeee31c3ed9dd0ba241368e690c))
* **dataset:** convert 3c label to 1c label ([09f71b7](https://github.com/tpnam0901/QTSeg/commit/09f71b7d4afd2232380a84aa5c205636bdd129cd))
* **dataset:** dataset by pass augmentation ([cde55eb](https://github.com/tpnam0901/QTSeg/commit/cde55eb04c5b745158eb514b15bbe4dd01fd08c8))
* **dataset:** pad image ([ddea7f9](https://github.com/tpnam0901/QTSeg/commit/ddea7f945d984431256f1efebe28c57b1b6f81e2))
* **dataset:** pad image DSB2018 ([58ec66e](https://github.com/tpnam0901/QTSeg/commit/58ec66e9bf3e2db75b82ee30dc8e7465241279c9))
* **dataset:** wrong params init ([accca62](https://github.com/tpnam0901/QTSeg/commit/accca62c224f8908d68afb830cf0711fae4529c0))
* **decoder:** remove unused code ([513c6de](https://github.com/tpnam0901/QTSeg/commit/513c6de282f2acc65a3fbb90012188ed68aee6b4))
* function not found ([bcfeebf](https://github.com/tpnam0901/QTSeg/commit/bcfeebfc397a73e9e9658e2b469b38e840b830cc))
* **infer:** 4 channels image ([395ab77](https://github.com/tpnam0901/QTSeg/commit/395ab7722f9e6c5d9fc4466617e2cd57c28b4587))
* **infer:** 4 channels image ([cef3436](https://github.com/tpnam0901/QTSeg/commit/cef3436e54c22eb92aa52ef63003d533e5fde3e8))
* **infer:** checkpoint is none in infer ([097ece0](https://github.com/tpnam0901/QTSeg/commit/097ece0b19f6a4193fdf09ef4b9a058e0476e9bb))
* **infer:** crop image and return original image ([0de1a6c](https://github.com/tpnam0901/QTSeg/commit/0de1a6cd693b23e5c20ee3dfce3a25daeee3c74d))
* **metrics:** can not calculate original size ([197d85a](https://github.com/tpnam0901/QTSeg/commit/197d85a28b71ac45672b488857c589e771364c7f))
* **MLFD:** rename model from MLFF to MLFD ([60635af](https://github.com/tpnam0901/QTSeg/commit/60635af95b999bcfec3eded8c94ca9c74ba086af))
* **SEED:** out of range ([042761b](https://github.com/tpnam0901/QTSeg/commit/042761b055cace44c758c7593ffc9297d13fc4b9))
* update sklearn from 1.3.2 -&gt; 1.5.0 ([bfb28af](https://github.com/tpnam0901/QTSeg/commit/bfb28af07cf9fb05524752cfbec9bcfc655c0f78))


### Documentation

* add pretrained ([e4cb92c](https://github.com/tpnam0901/QTSeg/commit/e4cb92c3e6f3307ccf44611b108ab116475c4787))
* fix pretrained link, release link ([446314b](https://github.com/tpnam0901/QTSeg/commit/446314b6ac42ef2c580bd48106b62cbb9ec069a0))
* fix release page link ([635ad8e](https://github.com/tpnam0901/QTSeg/commit/635ad8ed4ccc13e269df9fc4c3e799537addbc29))
* reduce image size of DMAD ([78d99d6](https://github.com/tpnam0901/QTSeg/commit/78d99d64e44535c642994f1e88e58e9bc414ca88))
* reduce image size of DMAD ([955c721](https://github.com/tpnam0901/QTSeg/commit/955c7218a999a141612e7771ec82d8f2bcff0849))
* update architecture overall ([7a0b3c8](https://github.com/tpnam0901/QTSeg/commit/7a0b3c86f76c1f07f9595ecf3f63b8bac1003519))
* update architecture overall, add fives, dsb2018 dataset, update viz ([da3e6c1](https://github.com/tpnam0901/QTSeg/commit/da3e6c1221aa4157129978d4a841d3ebd494cdfb))

## 0.1.0 (2024-12-24)


### Features

* add a copy of google colab ([d396dd3](https://github.com/tpnam0901/QTSeg/commit/d396dd3645ac9b18ac6925949e8b001ec3149bab))
* add base config ([4f779c6](https://github.com/tpnam0901/QTSeg/commit/4f779c672448fd20816a0653cef72a7903710c6d))
* add BKAI, BUSI, ISIC configs ([ec3ca3e](https://github.com/tpnam0901/QTSeg/commit/ec3ca3e30dbb5eb5c467dea0a3719b8816d7f7ae))
* add dataset, dataloader for BKAI, BUSI, ISIC ([829381a](https://github.com/tpnam0901/QTSeg/commit/829381ab0ebe03de0b15b05bf6a0e844b83a6985))
* add environment settings ([4ec3721](https://github.com/tpnam0901/QTSeg/commit/4ec3721010b8fae6fed94b3f0a7a126dc099b52d))
* add FPN Encoder module ([5ce738f](https://github.com/tpnam0901/QTSeg/commit/5ce738f8471e794d0e62c6029e7658bf275b8625))
* add loss function, metrics, lr schedulers ([607819c](https://github.com/tpnam0901/QTSeg/commit/607819cbfa1db3154275ad27408d4823ef54792b))
* add MQM Decoder module ([6fed401](https://github.com/tpnam0901/QTSeg/commit/6fed401b72f102f83fcd00e870f0e47acaf2fdb5))
* add optimizers custom ([406ba6e](https://github.com/tpnam0901/QTSeg/commit/406ba6e153724f8989cf56ff04f036ef1b3c4195))
* add preprocess tools for BKAI, BUSI ([823d4c7](https://github.com/tpnam0901/QTSeg/commit/823d4c727971d5be64b3fbe8759d9a10af60a41b))
* add QTSeg model ([7f8657d](https://github.com/tpnam0901/QTSeg/commit/7f8657d66341823100c51f57e69f2eed37d44330))
* add train, eval, infer scripts ([87e4304](https://github.com/tpnam0901/QTSeg/commit/87e430448051f4150ebb08ce1064618638505472))
* init repo ([0184d7c](https://github.com/tpnam0901/QTSeg/commit/0184d7cfd7c4fb97fcae9f151fb3b50c6a83d320))


### Bug Fixes

* BUSI, ISIC dataloader config ([17fdb96](https://github.com/tpnam0901/QTSeg/commit/17fdb965adcaa16ce7e843d8f8cfe1c45baea16a))
* wrong epoch increment ([cee022b](https://github.com/tpnam0901/QTSeg/commit/cee022b4578692b96b30e5f10e195e950e786506))


### Documentation

* add arxiv, citation ([7b663e8](https://github.com/tpnam0901/QTSeg/commit/7b663e8809eebee426a76ff34d253f0621a23248))
* add colab link ([fe69f84](https://github.com/tpnam0901/QTSeg/commit/fe69f8412478cd347034abd2cec3073860cc5785))
* add readme and preview image ([81ba765](https://github.com/tpnam0901/QTSeg/commit/81ba765295c9193c16a7c44c12fd7c838a48312d))
* fix BKAI, BUSI link ([ecf90c6](https://github.com/tpnam0901/QTSeg/commit/ecf90c656afca528b22be2d8747b5180b3c4f38c))
* fix title ([ea5c531](https://github.com/tpnam0901/QTSeg/commit/ea5c531d13ad06228311ef9f560be64aaa412ca3))
* rearrange figure position ([2e8448f](https://github.com/tpnam0901/QTSeg/commit/2e8448f2d1a2d18c84cbe38bdde4abd104f3dfeb))
* update abstract ([15b5131](https://github.com/tpnam0901/QTSeg/commit/15b51312c5ab6dad64bd2a36fa5d9701c615b8eb))
