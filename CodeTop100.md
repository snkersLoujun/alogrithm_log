1. 4月份至少刷完一遍，这是最基础的
   - 4/12开始每天至少4题 （长路漫漫呀）
2. 记得汇总重点题
    - 以下为重点题
        - LC146：LRU缓存机制
        - LC25：K个一组翻转链表
        - LC42: 接雨水
        - LC148:排序链表
        - LC143:重排链表
        - LC4:寻找两个正序数组的中位数
        - LC72:编辑距离



截至4/11 刷题数：26
截至4/17 刷题数：40


####  2022/4/1

##### `LC206`: 反转链表

```C++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head) return head;
        auto a = head , b = head->next;
        while(b){
            auto c = b->next;
            b->next = a;
            a = b , b = c ;
        }
        head->next = nullptr;
        return a ;
    }
};
```

##### `LC3`:无重复的最长子串

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<int ,int>hash;
        int res = 0 ;
        for(int i = 0 , j = 0  ; i < s.size() ; i++)
        {
            hash[s[i]]++;
            while(j < i && hash[s[i]] > 1) hash[s[j++]] --;
            res = max(res , i - j + 1); 
        }
        return res ;
    }
};

```



##### LC5:最长回文串

```C++
class Solution {
public:
    string longestPalindrome(string s) {
        string res ;
        for(int i = 0 ; i < s.size() ; i++){
            int l = i - 1 , r = i + 1;
            while(l >= 0 && r < s.size() && s[l] == s[r]) l-- , r++;
            if(res.size() < r - l - 1) res = s.substr(l + 1 , r - l - 1);

            l = i  , r = i + 1;
            while(l >= 0 && r < s.size() && s[l] == s[r]) l-- , r++;
            if(res.size() < r - l - 1) res = s.substr(l + 1 , r - l - 1);
        }
        return res ;
    }
};



```



#### 2022/4/3

>1. 算法今天再好好思考一下
>   - 快速排序
>   - 第K个数
>   - 归并排序
>   - 逆序队的数量
>   - LRU缓存机制
>2. 把`cmake`复习一下，彻底掌握
>
>

##### LC215：数组中最大的第K个元素

```C++
//小根堆的写法
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        priority_queue<int ,vector<int> , greater<int>>q;
        for(int i = 0 ; i < nums.size() ; i++){
            if(q.size() < k )
            {
                q.push(nums[i]);
            }
            else
            {
                if(q.top() <= nums[i]){
                    q.pop();
                    q.push(nums[i]);
                }
            }
        }
        return q.top();
    }
};

//快速排序写法
class Solution {
public:

    int quick_sort(vector<int>& nums , int l , int r , int k){
        if(l == r) return nums[l];
        int x = nums[(l+r)/2] , i = l -1 , j = r + 1;
        while(i < j){
            do i++ ; while(nums[i] > x);
            do j-- ; while(nums[j] < x);
            if(i < j) swap(nums[i] , nums[j]);
        }
        if( k <= j - l + 1 ) return quick_sort(nums , l , j , k);
        else return quick_sort(nums ,j + 1 , r , k - (j - l + 1)); 
    }
    int findKthLargest(vector<int>& nums, int k) {
        return quick_sort(nums , 0 , nums.size() - 1, k);
    }
};
```



##### LC146：LRU缓存机制（**重点**）

```c++
//很重要，反复看为主
class LRUCache {
public:
    struct Node {
        int key, val;
        Node *left, *right;
        Node(int _key, int _val): key(_key), val(_val), left(NULL), right(NULL) {}
    }*L, *R;
    unordered_map<int, Node*> hash;
    int n;

    void remove(Node* p) {
        p->right->left = p->left;
        p->left->right = p->right;
    }

    void insert(Node* p) {
        p->right = L->right;
        p->left = L;
        L->right->left = p;
        L->right = p;
    }

    LRUCache(int capacity) {
        n = capacity;
        L = new Node(-1, -1), R = new Node(-1, -1);
        L->right = R, R->left = L;
    }

    int get(int key) {
        if (hash.count(key) == 0) return -1;
        auto p = hash[key];
        remove(p);
        insert(p);
        return p->val;
    }

    void put(int key, int value) {
        if (hash.count(key)) {
            auto p = hash[key];
            p->val = value;
            remove(p);
            insert(p);
        } else {
            if (hash.size() == n) {
                auto p = R->left;
                remove(p);
                hash.erase(p->key);
                //delete p;
            }
            auto p = new Node(key, value);
            hash[key] = p;
            insert(p);
        }
    }
};

```



##### LC25：K个一组翻转链表（**重点**）

##### LC15：三数之和

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin() , nums.end());
        vector<vector<int>>res;
        for(int i = 0 ; i < nums.size() ; i++){
            if(i && nums[i] == nums[i-1]) continue;
            for(int j = i + 1, k = nums.size() - 1; j < k ; j++ ){
                if(j > i + 1 && nums[j] == nums[j-1]) continue;
                while(j < k - 1 && nums[i] + nums[j] + nums[k - 1] >= 0) k--;
                if(nums[i] + nums[j] + nums[k] == 0){
                    res.push_back({nums[i] , nums[j] , nums[k]});
                } 
            }
        }
        return res ;
    }
};
```



##### LC912:手撕快速排序

```c++
class Solution {
public:
    void quick_sort(vector<int>& nums , int l , int r){
        if(l >= r) return ;
        int i = l - 1 , j = r + 1  , x = nums[(l + r)/2];
        while(i < j){
            do i++ ; while(nums[i] < x);
            do j-- ; while(nums[j] > x);
            if(i < j) swap(nums[i] ,nums[j]);
        }
        quick_sort(nums , l , j);
        quick_sort(nums , j + 1 , r);

    }
    vector<int> sortArray(vector<int>& nums) {
    quick_sort(nums , 0 , nums.size() - 1);
    return nums;
    }
};
```



##### LC53:最大子序和

```C++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res = nums[0] ;
        vector<int>f(nums.size());
        f[0] = nums[0];
        for(int i = 1 ; i < nums.size() ; i++){
            f[i] = max(f[i-1] + nums[i] , nums[i]);
            res = max(res , f[i]);
        }
        return res;
    }
};
```



#### 2022/4/4

##### LC1:两数之和

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int ,int >hash;
        for(int i = 0 ; i < nums.size() ; i++){   
            int anoher = target - nums[i];
            if(hash.count(anoher)) return {hash[anoher] , i};
            else hash[nums[i]] = i ;
        }
        return {};
    }
};
```



##### LC21:合并两个有序链表

```C++
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        auto dummy = new ListNode(-1);
        auto tail = dummy;
        while(list1 && list2){
            if(list1->val <= list2->val){
                dummy->next = list1;
                dummy = dummy->next;
                list1 = list1->next;
            }
            else{
                dummy->next = list2;
                dummy = dummy->next;
                list2 = list2->next;
            }
        }
        if(list1) dummy->next = list1;
        if(list2) dummy->next = list2;
        return tail->next;
    }
};
```

#### 2022/4/5

##### LC102:二叉树的层序遍历

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>>res;
        queue<TreeNode*>q;
        if(!root) return res;
        q.push(root);
        while(!q.empty()){
            vector<int>level;
            int n = q.size();
            for(int i = 0 ; i < n ; i++){
                auto t = q.front();
                level.push_back(t->val);
                q.pop();
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
            }
            res.push_back(level);
        }
        return res ;
    }
};
```

##### LC141:环形链表

```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head || !head->next) return false;
        auto fast = head, slow = head;
        while(fast){
            fast = fast->next , slow = slow->next;
            if(fast) fast = fast->next;
            if(slow == fast) return true;
        }
        return false;
    }
};
```

##### LC121:买卖股票的最佳时机

```c++
//用的动态规划
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        int f[n][2];
        f[0][0] = 0 , f[0][1] = -prices[0];
        for(int i = 1 ; i < n ; i++){
            f[i][0] = max(f[i-1][0] , f[i-1][1] + prices[i]);
            f[i][1] = max(f[i-1][1] ,  - prices[i]);
        }
        return f[n-1][0];
    }
};
```

#### 2022/4/6

##### `LC103`: 二叉树的锯齿形层序遍历

```c++
//顶定义一个计数器，判断奇偶即可
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>>res;
        if(!root) return res;
        queue<TreeNode*>q;
        q.push(root);
        int cnt = 0 ;
        bool flag = true;
        while(!q.empty()){
            vector<int>level;
            int n = q.size();
            for(int i = 0 ; i < n ;i++){
                auto t = q.front();
                q.pop();
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
                level.push_back(t->val);
            }
            if(++cnt % 2 == 0 ) reverse(level.begin() , level.end());
            res.push_back(level);
        }
        return res;
    }
};
```

##### `LC20`:有效的括号

```c++
class Solution {
public:
    bool isValid(string s) {
    stack<char>stk;
    for(auto c : s){
        if(c == '(' || c == '{' || c == '['){
            stk.push(c);
        }
        else if(c == ')'){
            if(stk.empty() || stk.top() != '(')
                return false;
            stk.pop();
        }
        else if(c == '}'){
            if(stk.empty() || stk.top() != '{')
                return false;
            stk.pop();
        }
        else{
            if(stk.empty() || stk.top() != '[')
                return false;
            stk.pop();
        }
    }
    return stk.empty();
    }
};
```

#### 2022/4/7

##### LC88:合并两个有序数组

```c++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int k = m + n - 1, i = m - 1 , j = n - 1;
        while(i >= 0 && j >= 0){
            if(nums1[i] > nums2[j]) nums1[k--] = nums1[i--];
            else nums1[k--]=nums2[j--];
        }
        while(j >= 0 )nums1[k--] = nums2[j--];
    }
};
```



##### LC160:相交链表

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        auto a = headA, b = headB;
        while(a!=b){
            if(a) a = a->next ;
            else a = headB;
            if(b) b = b->next ;
            else b = headA;
        }
        return a;
    }
};
```



##### `LC33`:搜索旋转排序数组

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0 , r = nums.size() - 1;
        while(l < r)
        {
            int mid = l + r + 1 >> 1;
            if(nums[mid] >= nums[0]) l = mid;
            else r = mid - 1;
        }
        
        if(target >= nums[0]) l = 0 ;
        else l = r + 1, r = nums.size() - 1;
        while(l < r){
            int mid = l + r >> 1 ;
            if(nums[mid] >= target) r = mid;
            else l = mid + 1;
        }

        if(nums[r] == target) return r;//这里用l的话，特殊样例过不了，自己思考一下原因
        else return -1;
    }
};
```



附加题：每日一题

`LC113`:路径总和

```c++
class Solution {
public:
    vector<vector<int>>res;
    vector<int>level;
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if(root) dfs(root , targetSum);
        return res ;
    }

    void dfs(TreeNode* root , int targetSum){
        level.push_back(root->val);
        targetSum -= root->val;
        if(!root->left && !root->right && targetSum == 0) res.push_back(level); 
        else{
            if(root->left) dfs(root->left , targetSum);
            if(root->right) dfs(root->right , targetSum);
        }
        level.pop_back();
    }
};
```

#### 2022/4/11

##### `LC5`：最长回文子串

```c++
class Solution {
public:
    string longestPalindrome(string s) {
        string res ;
        for(int i = 0 ; i < s.size() ; i++){ 
            int l = i , r = i + 1;
            while(l >= 0 && r < s.size() && s[l] == s[r] ) l-- , r++;
            if(res.size() < r - l - 1) res = s.substr(l + 1 , r - l - 1);

            l = i - 1 , r = i + 1;
            while(l >= 0 && r < s.size() && s[l] == s[r] ) l-- , r++;
            if(res.size() < r - l - 1) res = s.substr(l + 1 , r - l - 1);
        }
        return res ;
    }
};
```



##### `LC236`: 二叉树的最近公共祖先

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root || p == root || q == root) return root;
        auto left = lowestCommonAncestor(root->left , p , q);
        auto right = lowestCommonAncestor(root->right , p , q);
        if(!left) return right;
        if(!right) return left;
        return root ; 
    }
};
```



##### `LC200`：岛屿数量

```c++
//真是任重而道远
class Solution {
public:
    vector<vector<char>>g;
    int dx[4] = {0 , 1 , 0 , -1} ,dy[4] = {-1 , 0 , 1 , 0};
    int numIslands(vector<vector<char>>& grid) {
        g = grid;
        int cnt = 0 ;
        for(int i = 0 ; i < g.size() ; i++)
            for(int j = 0 ; j < g[i].size() ; j++)
                if(g[i][j] == '1'){
                    dfs(i , j);
                    cnt++;
                }
        return cnt;
    }

    void dfs(int x , int y){
        g[x][y] = 0 ;
        for(int i = 0 ; i < 4 ; i++){
            int a = x +dx[i] , b = y + dy[i];
            if(a >= 0 && a < g.size() && b>= 0 && b < g[a].size() && g[a][b] == '1')
            dfs(a ,b);
        }
    }
};
```



##### `LC46`：全排列

```c++
//感觉中邪了
class Solution {
public:
    vector<vector<int>> ans;
    vector<bool> st;
    vector<int> path;

    vector<vector<int>> permute(vector<int>& nums) {

        st = vector<bool>(nums.size() , false);
        dfs(nums, 0);
        return ans;
    }

    void dfs(vector<int> &nums, int u)
    {
        if (u == nums.size())
        {
            ans.push_back(path);
            return ;
        }

        for (int i = 0; i < nums.size(); i ++ ){
            if (!st[i])
            {
                st[i] = true;
                path.push_back(nums[i]);
                dfs(nums, u + 1);
                st[i] = false;
                path.pop_back();
            }
        }
    }
};
```



##### `LC415`：字符串相加

```c++
class Solution {
public:
    vector<int> add(vector<int>&A , vector<int>&B){
        vector<int>C;
        for(int i = 0 , t = 0 ; i < A.size() || i < B.size() || t ;i++){
            if(i < A.size()) t += A[i];
            if(i < B.size()) t += B[i];
            C.push_back(t%10);
            t /= 10 ;
        }
        return C;
    }

    string addStrings(string num1, string num2) {
    vector<int>A , B;
    for(int i = num1.size() - 1 ; i >= 0  ; i--) A.push_back(num1[i] - '0');  
    for(int i = num2.size() - 1 ; i >= 0  ; i--) B.push_back(num2[i] - '0'); 
    string c ;
    auto C = add(A , B);
    for(int i = C.size() - 1; i >= 0 ; i--) c += to_string(C[i]); 
    return c ;
    }
};
```

##### `LC141`: 环形链表I

```C++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head || !head->next) return false;
        auto fast = head, slow = head;
        while(fast){
            fast = fast->next , slow = slow->next;
            if(fast) fast = fast->next;
            if(slow == fast) return true;
        }
        return false;
    }
};
```



##### `LC142`: 环形链表II

```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        auto fast = head , slow = head;
        
        while(fast){
            fast = fast->next , slow = slow->next;
            if(!fast) return NULL;
            fast = fast->next;

            if(fast == slow){
                slow = head;
                while(slow != fast) fast = fast->next , slow = slow->next;
                return fast;
            }
        }
        return NULL;
    }
};
```

最关键的是每一题自己理解并且想通

#### 2022/4/14

#####  ` LC23`：合并K个排序链表(标记)

```c++
```



#####   `LC92`：反转链表II

```c++
//不应该出现的情况
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        auto dummy = new ListNode(-1);
        dummy->next = head;
        auto a = dummy  ;
        for(int i = 0 ; i < left - 1  ; i++) a = a->next;
        auto b = a->next  , c = b->next;
        for(int i = 0 ; i < right - left ; i++){
            auto d = c->next;
            c->next = b ;
            b = c , c = d;
        }
        a->next->next = c;
        a->next = b;
        return dummy->next;
    }
};
```



#####   `LC54`：螺旋矩阵

```c++
//真是很经典了
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int>res;
        int n = matrix.size() , m = matrix[0].size();
        if(!n) return res;
        vector<vector<bool>>st(n , vector<bool>(m));
        int dx[4] = {0 , 1 , 0 , -1 } , dy[4] = {1 , 0 , -1 , 0};
        
        for(int i = 0 , x = 0 , y = 0 , d = 0 ; i < m *n ; i++){
            res.push_back(matrix[x][y]);
            st[x][y] = true;
            int a = x + dx[d] , b = y + dy[d];
            if(a < 0 || a >= n || b < 0 || b >= m || st[a][b]){
                d = (d + 1)%4;
                a = x + dx[d] , b = y + dy[d];
            }
            x = a , y = b ;
        }
        return res ;
    }
};
```



#####   `LC300`：最长上升子序列

```c++
//笨也确实贼笨
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        if(nums.size() == 1) return nums.size();
        vector<int>f(nums.size() , 1);
        f[0] = 1;
        int res = 0 ; 
        for(int i = 1 ; i < nums.size() ; i++){
            for(int j = 0 ; j < i ; j++){
                if(nums[j] < nums[i]) f[i] = max(f[j] + 1 , f[i]);
                res = max(res , f[i]);
            }
        }
        return res ;
    }
};
```



#####   `LC42`: 接雨水（**重点**）

```c++
```



##### `LC704`: 二分查找

```c++
//秒杀题
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0 , r = nums.size() - 1;
        while( l < r){
            int mid = l + r >> 1 ;
            if(nums[mid] >= target) r = mid ; 
            else l = mid + 1;
        }
        if(nums[l] == target) return l ;
        else return -1;
    }
};
```

##### `LC148`:排序链表(重点)

```c++
//头晕中
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) n ++ ;

        for (int i = 1; i < n; i *= 2) {
            auto dummy = new ListNode(-1), cur = dummy;
            for (int j = 1; j <= n; j += i * 2) {
                auto p = head, q = p;
                for (int k = 0; k < i && q; k ++ ) q = q->next;
                auto o = q;
                for (int k = 0; k < i && o; k ++ ) o = o->next;
                int l = 0, r = 0;
                while (l < i && r < i && p && q)
                    if (p->val <= q->val) cur = cur->next = p, p = p->next, l ++ ;
                    else cur = cur->next = q, q = q->next, r ++ ;
                while (l < i && p) cur = cur->next = p, p = p->next, l ++ ;
                while (r < i && q) cur = cur->next = q, q = q->next, r ++ ;
                head = o;
            }
            cur->next = NULL;
            head = dummy->next;
        }

        return head;
    }
};

```

`LC143`:重排链表(重点)

```c++
//不是很好理解
class Solution {
public:
    void reorderList(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) n++;

        if (n <= 2) return;

        int half = n / 2;
        auto a = head;
        for (int i = 0; i < half; i++) a = a->next;

        auto b = a->next;  // 此行和下一行的顺序不能换
        a->next = nullptr; // 此时a是尾结点，next赋为空
        while (b){
        // for (int i = 0; i < n - half - 1; i++){
            auto c = b->next;
            b->next = a;
            a = b, b = c;
        }

        // 注意，此处循环 (n-1)/2 次
        for (int i = 0; i < (n - 1) / 2; i++){
            b = a->next;
            a->next = head->next;
            head->next = a;
            head = a->next;
            a = b;
        }
    }
};

```

#### `LC94:二叉树的中序遍历`
```c++
//仅仅掌握递归写法远远不够
class Solution {
public:
    vector<int>res;
    vector<int> inorderTraversal(TreeNode* root) {
        if(!root) return res;
        dfs(root);
        return res;
    }

    void dfs(TreeNode* root){
        if(!root) return ;
        dfs(root->left);
        res.push_back(root->val);
        dfs(root->right);
        
    }
};

//必须掌握迭代写法(更新中···)
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int>res;
        stack<TreeNode*>stk;
        while(root || stk.size()){
            while(root){
                stk.push(root);
                root = root->left;
            }

            root = stk.top();
            stk.pop();
            res.push_back(root->val);
            root = root->right;
        }
        return res ;
    }
};
```
#### LC124:二叉树中的最大路径和
```c++
class Solution {
public:
    int ans;
    int maxPathSum(TreeNode* root) {
        ans = INT_MIN;
        dfs(root);
        return ans ;
    }

    int dfs(TreeNode* root){
        if(!root) return 0 ;
        int left = max(0 , dfs(root->left)) , right = max(0 , dfs(root->right));

        ans = max(ans , left + right + root->val);
        return root->val + max(left , right);
    }
};
```

#### LC232:用栈实现队列
```c++
class MyQueue {
public: 
    stack<int>stk1 , stk2 ;
    MyQueue() {

    }
    
    void push(int x) {
        stk1.push(x);
    }
    
    int pop() {
        while(!stk1.empty()){
            stk2.push(stk1.top());
            stk1.pop();
        }
        auto t = stk2.top();
        stk2.pop();
        while(!stk2.empty()){
            stk1.push(stk2.top());
            stk2.pop();
        }
        return t;
    }
    
    int peek() {
        while(!stk1.empty()){
            stk2.push(stk1.top());
            stk1.pop();
        }
        auto t = stk2.top();
        while(!stk2.empty()){
            stk1.push(stk2.top());
            stk2.pop();
        }
        return t ;
    }
    
    bool empty() {
        return stk1.empty();
    }
};
```

#### LC199:二叉树的右视图
```c++
class Solution {
public:
    queue<TreeNode*>q;
    vector<int>res;
    vector<int> rightSideView(TreeNode* root) {
        if(!root) return res;
        q.push(root);
        while(q.size()){
            vector<int>level;
            int n = q.size();
            while(n--){
                auto t = q.front();
                q.pop();
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
                level.push_back(t->val);
            }
            res.push_back(level[level.size() - 1]);
        }
        return res;
    }
};
```

#### lc70:爬楼梯
```c++
class Solution {
public:
    int climbStairs(int n) {
        long long  a = 1 , b = 2 ;
        while(--n){
            long long c = a + b ;
            a = b , b = c ;
        }
        return a ;
    }
};
```

#### LC19:删除链表的倒数第n个节点
```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        auto dummy = new ListNode(-1);
        dummy->next = head ;
        auto a = dummy;
        int cnt = 0 ;
        for(auto p = head; p ; p = p->next ) cnt++;
        for(int i = 0 ; i < cnt - n ; i++){
            a = a->next;
        }
        a->next = a->next->next;  
        return dummy->next;
    }
};
```
#### lc56:合并区间
```c++
//很经典的题目
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& a) {
        vector<vector<int>>res;
        if(a.empty()) return res;
        sort(a.begin() , a.end());
        int l = a[0][0] , r = a[0][1];

        for(int i = 1 ; i < a.size() ; i++){
            if(a[i][0] > r){
                res.push_back({l , r});
                l = a[i][0] , r = a[i][1];
            }else r = max(r , a[i][1]);
            
        }
        res.push_back({l , r});
        return res;
    }
};
```

#### lc4:寻找两个正序数组的中位数(重点)
```C++


```
#### LC82:删除排序链表中的重复元素II
```c++
//脑子有点乱了
//删除所有重复元素
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
    auto dummy = new ListNode(-1);
    dummy->next = head;
    auto p = dummy;
    while(p->next){
        auto q = p->next->next;
        while(q && p->next->val == q->val) q = q->next;
        if(p->next->next == q) p = p->next;
        else p->next = q;
    }
    return dummy->next;
    }
};


//母题：保留其中的一个重复元素(LC:83)
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(!head) return nullptr;
        ListNode* p = head;
        while(p->next){
            if(p->val == p->next->val) p->next = p->next->next;
            else p = p->next;
        }
        return head;
    }
};
```
#### LC2:两数相加
```c++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        auto dummy = new ListNode(-1) , cur = dummy;
        int t = 0 ;
        while(l1 || l2 || t){
            if(l1) t += l1->val , l1 = l1->next;
            if(l2) t += l2->val , l2 = l2->next;
            cur = cur->next = new ListNode(t%10);
            t /= 10;
        }
        return dummy->next;
    }
};
```

#### LC8:字符串转换整数
```C++
//其实是简单的模拟题
class Solution {
public:
    int myAtoi(string s) {
        int k = 0 ;
        while(k < s.size() && s[k] == ' ') k++;
        if(k == s.size()) return 0 ;

        int minus = 1;
        if(s[k] == '-') minus = -1,k++;
        else if(s[k] == '+') k++;

        long long res = 0;
        while(k < s.size() && s[k]>= '0'&& s[k] <= '9'){
            res = res * 10 + s[k] - '0';
            k++;
            if(res > INT_MAX) break;
        }
        res *= minus;
        if(res > INT_MAX) res = INT_MAX;
        if(res < INT_MIN) res = INT_MIN;
        return res;

    }
};
```

#### LC41：缺失的第一个正数
#### LC1143：最长上升子序列

#### LC22: 括号生成

#### LC144:二叉树的前序遍历
```c++
//递归写法
class Solution {
public:
    vector<int>ans;
    vector<int> preorderTraversal(TreeNode* root) {
        dfs(root);
        return ans;
    }

    void dfs(TreeNode* root){
        if(!root) return ;
        ans.push_back(root->val);
        dfs(root->left);
        dfs(root->right);
    }
};

//迭代写法


```
