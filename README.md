<!--
* Graph-Based Learning from Demonstration Framework
*
* Copyright (c) 2024 Natalia Quiroga Perez. All rights reserved.
*
* This software may be modified and distributed
* under the terms of the BSD 3-Clause license.
*
* Refer to the LICENSE file for details.
*
-->

<h1>Learning from Demonstration<sup>®</sup> Framework</h1>

<a id="markdown-description" name="description"></a>
# Description

The official repository contains documentation and datasets for QTrobot, NAO and Kinova gen3. This Framework runs in Python.
The repository has been tested on Ubuntu 20.04.

<h1>Table of Contents</h1>

<!-- TOC -->

- [Description](#description)
- [Licensing](#licensing)
- [Reference](#reference)
      - [Useful Links](#useful-links)

<!-- /TOC -->

<a id="markdown-licensing" name="licensing"></a>
# Licensing 
This repository is licenced under the [BSD 3-Clause "Revised" License](./LICENSE) 

<a id="markdown-role-of-google-protobuf-in-kortex-api" name="role-of-google-protobuf-in-kortex-api"></a>
# Role of Google Protocol Buffer in Kortex API 

The Kortex API uses Google Protocol Buffer message objects<sup>**[1](#useful-links)**</sup> to exchange data between client and server.  

Google Protocol Buffer offers structured data objects with standard methods for each member field:  
+ structured, nested objects
+ basic types and collections
+ getter/setter methods on basic types
+ iterators, dimension and appending methods on collections
+ many helpers (e.g. serialize/deserialize, I/O functions)
  

<a id="markdown-quick-start-howto-python" name="quick-start-howto-python"></a>
## Quick Start 

  To run the Python examples you will need to install the Python interpreter and the pip installation module.

  Here is some general information about the Python interpreter and the pip module manager.  
  - [Error management](./linked_md/python_error_management.md)
  - [Examples](./api_python/examples/readme.md)

# Reference
#### Useful Links
|  |  |
| ---: | --- |
| Kinova home page: | [https://www.kinovarobotics.com](https://www.kinovarobotics.com)|
| Google Protocol Buffers home page: | [https://developers.google.com/protocol-buffers](https://developers.google.com/protocol-buffers) |
