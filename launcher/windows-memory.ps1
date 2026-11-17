$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PrefillMemory
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Info
    {
        public uint Size;
        public UIntPtr CommitTotal, CommitLimit, CommitPeak, PhysicalTotal, PhysicalAvailable;
        public UIntPtr SystemCache, KernelTotal, KernelPaged, KernelNonPaged, PageSize;
        public uint Handles, Processes, Threads;
    }

    [DllImport("psapi.dll", SetLastError=true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetPerformanceInfo(ref Info value, uint size);

    public static string Sample()
    {
        var value = new Info();
        value.Size = (uint)Marshal.SizeOf(typeof(Info));
        if (!GetPerformanceInfo(ref value, value.Size))
            throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        ulong size = value.PageSize.ToUInt64();
        ulong available = value.PhysicalAvailable.ToUInt64() * size / 1048576;
        return available + "," + value.CommitTotal.ToUInt64() * size
            + "," + value.CommitLimit.ToUInt64() * size;
    }
}
'@
while ($true) {
    [Console]::WriteLine([PrefillMemory]::Sample())
    Start-Sleep -Milliseconds 350
}
