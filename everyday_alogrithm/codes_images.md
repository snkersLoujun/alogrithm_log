### 动态规划
#### 子序列问题
#### 2022/3/27

1.LC647回文子串

![LC647](https://raw.githubusercontent.com/snkersLoujun/Map-Bed/main/Snipaste_LC647.png)



```C++
class Solution {
public:
    int countSubstrings(string s) {
        int res = 0 ;
        for(int i = 0 ; i < s.size() ; i++){
            //枚举长度为奇数的情况
            for(int j = i , k = i ; j>= 0 && k <s.size() ; j--, k++){
                if(s[j]!= s[k]) break;
                res ++;
            }
            //枚举长度为偶数的情况
            for(int j = i , k = i + 1 ; j >= 0 && k < s.size() ; j-- , k++){
                if(s[j]!=s[k]) break;
                res ++;
            }
        }
        return res ;
    }
};


```
2.LC516最长回文子序列
```C++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        vector<vector<int>> dp(s.size(), vector<int>(s.size(), 0));
        for (int i = 0; i < s.size(); i++) dp[i][i] = 1;
        for (int i = s.size() - 1; i >= 0; i--) {
            for (int j = i + 1; j < s.size(); j++) {
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][s.size() - 1];
    }
};
```
***
#### 2022/3/28
1.LC392判断子序列
```C++
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int k = 0 ;
        for(int i = 0 ; i < t.size() ; i++){
            if(k < s.size() && s[k] == t[i]) k++;
        }
        return k == s.size();
    }
};
```

2.LC115不同的子序列
```C++
class Solution {
public:
    int numDistinct(string s, string t) {
        int n = s.size(), m = t.size();
        s = ' ' + s, t = ' ' + t;
        vector<vector<unsigned long long>> f(n + 1, vector<unsigned long long>(m + 1));
        for (int i = 0; i <= n; i ++ ) f[i][0] = 1;
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ ) {
                f[i][j] = f[i - 1][j];
                if (s[i] == t[j]) f[i][j] += f[i - 1][j - 1];
            }
        return f[n][m];
    }
};
```

3.LC583两个字符串的删除操作
```C++
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1));
        for (int i = 1; i <= n; i ++ ) f[i][0] = i;
        for (int i = 1; i <= m; i ++ ) f[0][i] = i;
        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ ) {
                f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
                if (word1[i - 1] == word2[j - 1])
                    f[i][j] = min(f[i][j], f[i - 1][j - 1]);
            }
        return f[n][m];
    }
};


```

4.LC72编辑距离
```C++
class Solution {
public:
    int minDistance(string a, string b) {
        int n = a.size(), m = b.size();
        a = ' ' + a, b = ' ' + b;
        vector<vector<int>> f(n + 1, vector<int>(m + 1));

        for (int i = 0; i <= n; i ++ ) f[i][0] = i;
        for (int i = 1; i <= m; i ++ ) f[0][i] = i;

        for (int i = 1; i <= n; i ++ )
            for (int j = 1; j <= m; j ++ ) {
                f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
                int t = a[i] != b[j];
                f[i][j] = min(f[i][j], f[i - 1][j - 1] + t);
            }

        return f[n][m];
    }
};

```
1.晚上写一个详细的计划
- 关于算法
- 关于八股文
- 关于项目
- 关于秋招

一切都要好好加油！！！


