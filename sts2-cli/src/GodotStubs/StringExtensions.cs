namespace Godot;

public static class StringExtensions
{
    // Preserve Godot resource/user prefixes rather than using OS path semantics.
    public static string PathJoin(this string instance, string file)
    {
        if (instance.Length == 0) return file;
        if (instance.EndsWith('/') || file.StartsWith('/')) return instance + file;
        return instance + "/" + file;
    }
}
