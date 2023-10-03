using System;

namespace CSErrorModel{
    public static class Utils
{
    public static T[] CreateArray<T>(int length, Func<int, T> f)
    {
        if (length == 0) return Array.Empty<T>();
        var array = new T[length];
        for (int i = 0; i < array.Length; i++) array[i] = f(i);
        return array;
    }

    public static T[] SubArray<T>(this T[] array, int index, int n)
    {
        var result = new T[n];
        Array.Copy(array, index, result, 0, n);
        return result;
    }

    public static void SetPropertyValue<T>(object obj, string propertyName, T value) => obj.GetType().GetProperty(propertyName).SetValue(obj, value);
}
}

