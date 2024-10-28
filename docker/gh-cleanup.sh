#!/bin/sh

echo Initial
df -h /
df /

FOLDERS="
    $AGENT_TOOLSDIRECTORY
    /opt/hostedtoolcache
    /opt/google/chrome
    /opt/microsoft/msedge
    /opt/microsoft/powershell
    /opt/pipx
    /usr/lib/mono
    /usr/local/julia*
    /usr/local/lib/android
    /usr/local/lib/node_modules
    /usr/local/share/chromium
    /usr/local/share/powershell
    /usr/share/dotnet
    /usr/share/swift
"

for folder in $FOLDERS ; do
    echo "Cleaning up $folder"
    sudo rm -rf $folder
    df /
done

echo "Final"
df -h /
