<!--
* Learning from Demonstration Framework <sup>®</sup>
*
* Copyright (c) 2018 Natalia Quiroga Perez. All rights reserved.
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

<a id="markdown-api-download-links" name="api-download-links"></a>
# Download links

The latest download links for each arm type are reported in the table below:

| Arm type       | Firmware     | Release notes      | API |
| :------------- | :----------: | :-----------: | :-----------:|
| Gen3 | [2.5.2](https://artifactory.kinovaapps.com:443/artifactory/generic-public/kortex/gen3/2.5.2/Gen3-2.5.2.swu) | [2.5.2](https://artifactory.kinovaapps.com:443/artifactory/generic-public/Documentation/Gen3/Technical%20documentation/Release%20notes/EN-eRN-001-Gen3-Ultralight-release-notes.pdf)    | [2.6.0](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.6.0)|
| Gen3 lite   | [2.3.2](https://artifactory.kinovaapps.com:443/artifactory/generic-public/kortex/gen3Lite/2.3.2/Gen3-lite-2.3.2.swu) | [2.3.2](https://artifactory.kinovaapps.com:443/artifactory/generic-public/Documentation/Gen3%20lite/Technical%20documentation/Release%20Notes/Gen3_lite_Release_Notes_2_3_2%20-%20R01.pdf) | [2.3.0](https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.3.0)|

When following the link to Artifactory, to download the correct C++ API, you have to select the package for your architecture on the left-hand side tree view and then click Download on the right-hand side:

 ![Artifactory](./linked_md/artifactory.png)

<details><summary>Previous releases</summary>
<p>
<ul>
<li>
Release 2.3.2 for Gen3 lite: <a href="https://artifactory.kinovaapps.com:443/artifactory/generic-public/kortex/gen3Lite/2.3.2/Gen3-lite-2.3.2.swu">Firmware</a>, <a href="https://artifactory.kinovaapps.com/ui/repos/tree/General/generic-public/kortex/API/2.3.0">Kortex API</a>
</li>
<li>
</li>
</ul>
</p>
</details>
<a id="markdown-build-and-run-instructions" name="build-and-run-instructions"></a>

# Build and Run instructions

[C++ API](./api_cpp/examples/readme.md)  
[Python API](./api_python/examples/readme.md) 

# Reference
#### Useful Links
|  |  |
| ---: | --- |
| Kinova home page: | [https://www.kinovarobotics.com](https://www.kinovarobotics.com)|
| Google Protocol Buffers home page: | [https://developers.google.com/protocol-buffers](https://developers.google.com/protocol-buffers) |
