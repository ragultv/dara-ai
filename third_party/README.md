# Vendored dependencies

| Library | Version | Source | SHA-256 of file | License |
| --- | --- | --- | --- | --- |
| nlohmann/json (`nlohmann/json.hpp`) | 3.12.0 | https://github.com/nlohmann/json/blob/v3.12.0/single_include/nlohmann/json.hpp | AAF127C04CB31C406E5B04A63F1AE89369FCCDE6D8FA7CDDA1ED4F32DFC5DE63 | MIT (`nlohmann/LICENSE.MIT`) |

Why vendored: the vcpkg manifest install needs tool downloads from GitHub release assets
(CMake 4.4.4, PowerShell 7.6.9), which returned HTTP 504 on the dev machine. vcpkg itself is
installed at D:\tools\vcpkg and `vcpkg.json` pins its baseline
(`ec48c11eae9448c8c6e7739d93e06fa2433aa95f`), so moving back is a one-line change.
