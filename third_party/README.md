# Vendored dependencies

| Library | Version | Source | SHA-256 of file | License |
| --- | --- | --- | --- | --- |
| nlohmann/json (`nlohmann/json.hpp`) | 3.12.0 | https://github.com/nlohmann/json/blob/v3.12.0/single_include/nlohmann/json.hpp | AAF127C04CB31C406E5B04A63F1AE89369FCCDE6D8FA7CDDA1ED4F32DFC5DE63 | MIT (`nlohmann/LICENSE.MIT`) |
| cpp-httplib (`httplib/httplib.h`) | 0.18.3 | https://github.com/yhirose/cpp-httplib/blob/v0.18.3/httplib.h | A0A0C13DC086663863DBE6730A19716F7D3744904B6205496E8301750814DBD5 | MIT (`httplib/LICENSE`) |
| SQLite amalgamation (`sqlite/sqlite3.c`, `sqlite3.h`) | 3.45.3 | https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip (zip SHA-256 EA170E73E447703E8359308CA2E4366A3AE0C4304A8665896F068C736781C651) | see the zip | Public domain |

Why vendored: the vcpkg manifest install needs tool downloads from GitHub release assets
(CMake 4.4.4, PowerShell 7.6.9), which returned HTTP 504 on the dev machine. vcpkg itself is
installed at D:\tools\vcpkg and `vcpkg.json` pins its baseline
(`ec48c11eae9448c8c6e7739d93e06fa2433aa95f`), so moving back is a one-line change.
