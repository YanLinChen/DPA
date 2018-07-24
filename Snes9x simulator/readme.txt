因為模擬器版權問題:此處只放執行檔
並且不放置遊戲ROM

使用方法步驟為
1.下載Snes9x模擬器的Source Code https://github.com/snes9xgit/snes9x
2.下載libpng, zlib這兩個library並且建置對應平台及位元數(我們使用VS2013 WIN32)
3.下載我們的Source Code並且建置對應平台及位元數(我們使用VS2013 WIN32)
4.在模擬器專案加入以上3個Library(步驟2,3)
5.取代模擬器原生Render.cpp為我們提供的即可。

或者下載https://github.com/YanLinChen/snes9x
再作步驟2~4。
