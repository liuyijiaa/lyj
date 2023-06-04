package main

import (
	"bytes"
	"crypto/rand"
	"crypto/sha1"
	"fmt"
	"math/big"
	rand1 "math/rand"
	"sort"
	"strconv"
)

const (
	alphaNum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	strLen   = 5 // 最小字符串长度
)

type kBucketTreeNode struct {
	nodeID  []byte             // 节点 ID
	value   map[string][]byte  // 节点值
	left    *kBucketTreeNode   // 左子节点
	right   *kBucketTreeNode   // 右子节点
	kBucket []*kBucketTreeNode // k-bucket 列表
}

type kBucketTree struct {
	root *kBucketTreeNode // 根节点
	k    int              // k-bucket 列表的 k 值，即每个节点维护的节点数量
}

//初始化k_bucket二叉树
func InitkBucketTree(k int, nodeId []byte) *kBucketTree {
	return &kBucketTree{
		root: &kBucketTreeNode{
			nodeID:  nodeId,
			value:   map[string][]byte{},
			left:    nil,
			right:   nil,
			kBucket: make([]*kBucketTreeNode, 0),
		},
		k: k,
	}
}

//添加节点
func (kBucketTree *kBucketTree) addNode(node *kBucketTreeNode) {
	//如果桶存在就更新桶
	if kBucketTree.updateNode(node) {
		kBucketTree.updateNode(node)
		return
	}
	//否则分裂桶
	dist := xorDistance(kBucketTree.root.nodeID, node.nodeID)
	tmp := kBucketTree.root //索引指针
	i := 0
	for i = 0; i < dist && tmp.right != nil; i++ {
		tmp = tmp.right
	}
	//判断桶满没有，如果没满就添加进去
	if len(tmp.kBucket) < kBucketTree.k {
		tmp.kBucket = append(tmp.kBucket, node)
		return
	} else {
		//桶的分裂操作
		//桶内排序
		tmp.kBucket = append(tmp.kBucket, node)
		sort.Slice(tmp.kBucket, func(i, j int) bool {
			return string(tmp.kBucket[i].nodeID) < string(tmp.kBucket[j].nodeID)
		})
		tmp1 := tmp
		min1 := 160                                  //分裂几层
		nodeIdArray := make([]int, len(tmp.kBucket)) //记录节点应属于的桶的编号
		for z, r := range tmp.kBucket {
			nodeIdArray[z] = xorDistance(r.nodeID, kBucketTree.root.nodeID)
			min1 = min(min1, xorDistance(r.nodeID, kBucketTree.root.nodeID))
		}
		min1 = min1 - i
		//持续分裂桶
		for z := 0; z < min1; z++ {
			tmp.left = new(kBucketTreeNode)
			tmp.right = new(kBucketTreeNode)
			tmp = tmp.right
		}
		//找到其中应该划分到另一桶的节点
		tmp.left = new(kBucketTreeNode)
		tmp.right = new(kBucketTreeNode)
		//节点重新进行归化桶
		for i1, r := range nodeIdArray {
			if r == min1+i {
				tmp.left.kBucket = append(tmp.left.kBucket, tmp1.kBucket[i1])
				if len(tmp.left.kBucket) == kBucketTree.k+1 {
					tmp.left.kBucket = tmp.left.kBucket[1:]
				}
			} else {
				tmp.right.kBucket = append(tmp.right.kBucket, tmp1.kBucket[i1])
			}
		}
		//将原先的桶置为空
		tmp1.kBucket = nil
	}
}

//更新操作，若桶未满假如桶中，桶满替换第一个节点（最老的节点）
func (kBucketTree *kBucketTree) updateNode(node *kBucketTreeNode) bool {
	belong := kBucketTree.findKBucket(node.nodeID)
	tmp := kBucketTree.root
	if belong != -1 {
		for i := 0; i < belong-1; i++ {
			tmp = tmp.right
		}
		if tmp.left != nil {
			tmp = tmp.left
		}
		if len(tmp.kBucket) < kBucketTree.k {
			tmp.kBucket = append(tmp.kBucket, node)
		} else {
			tmp.kBucket = tmp.kBucket[1:]
			tmp.kBucket = append(tmp.kBucket, node)
		}
		return true
	}
	return false

}

//用于查找节点所在的桶是否存在，如果存在返回桶编号，如果不存在返回-1
func (kBucketTree *kBucketTree) findKBucket(nodeId []byte) int {
	dis := xorDistance(nodeId, kBucketTree.root.nodeID)
	tmp := kBucketTree.root
	i := 0
	for i = 0; i < dis && tmp.right != nil; i++ {
		tmp = tmp.right
	}
	if i != dis || tmp.left == nil {
		return -1
	}
	return dis
}

//删除对应节点
func (receiver *kBucketTree) removeNode(nodeId []byte) {
	dist := xorDistance(nodeId, receiver.root.nodeID)
	tmp := receiver.root
	for i := 0; i < dist-1; i++ {
		if tmp.right == nil {
			break
		}
		tmp = tmp.right
	}
	if tmp.left != nil {
		for i := 0; i < receiver.k && tmp.left.kBucket != nil; i++ {
			if bytes.Equal(tmp.left.kBucket[i].nodeID, nodeId) {
				tmp.left.kBucket = append(tmp.left.kBucket[:i], tmp.left.kBucket[i+1:]...)
				return
			}
		}
	}
	if tmp.right.kBucket != nil {
		tmp = tmp.right
	}
	for i := 0; i < receiver.k && tmp.kBucket != nil; i++ {
		if bytes.Equal(tmp.kBucket[i].nodeID, nodeId) {
			tmp.kBucket = append(tmp.kBucket[:i], tmp.kBucket[i+1:]...)
			return
		}
	}

}

//节点异或运算
func xorDistance(id1, id2 []byte) int {
	distance := 0
	for i := 0; i < len(id1) && i < len(id2); i++ {
		tmp := id1[i] ^ id2[i]
		if tmp == uint8(0) {
			distance += 8
			continue
		}
		str := strconv.FormatInt(int64(tmp), 2)
		str = fmt.Sprintf("%08s", str) //转化为2进制
		for i2, r := range str {
			if r == '1' {
				return distance + i2
			}
		}
	}
	return distance
}

//打印已生成的桶中的节点
func (receiver *kBucketTree) print() {
	tmp := receiver.root
	for i := 0; tmp != nil; i++ {
		if tmp.left.kBucket == nil {
			fmt.Println("kBucket"+strconv.Itoa(160-i)+":", "[]")
		} else {
			for z := 0; z < receiver.k && tmp.left.kBucket != nil; z++ {
				if z < len(tmp.left.kBucket) && tmp.left.kBucket[z] != nil {
					fmt.Println("kBucket"+strconv.Itoa(160-i)+":", tmp.left.kBucket[z].nodeID)
				}
			}
		}

		tmp = tmp.right
		if tmp.left == nil {
			for z := 0; z < len(tmp.kBucket); z++ {
				if tmp.kBucket[z] == nil {
					break
				}
				fmt.Println("kBucket"+":", tmp.kBucket[z].nodeID)
			}
			return
		}
	}
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

//随机生成[]bytes
func generateBytes20() [20]byte {
	var id [20]byte
	_, err := rand.Read(id[:])
	if err != nil {
		panic(err)
	}
	return id
}

//hash计算字符串
func hashnodeKey(str string) [20]byte {
	return sha1.Sum([]byte(str))
}
//SetValue操作对树进行插入键值对操作
func (k *kBucketTreeNode) SetValue(key, value1 []byte) bool {
	//计算key的hash值，如果当前传入的value1是否为key的hash，不是则直接返回false
	keyHash := hashnodeKey(string(key))
	key1 := keyHash[:]
	if !bytes.Equal(key1, value1) {
		return false
	}
	if _, ok := k.value[string(key)]; ok {
		return true
	}
	k.value[string(key)] = value1
	k1 := kBucketTree{root: k}
	dist := k1.findKBucket(key)
	tmp := k
	i := 0
	for i = 0; i < dist-1 && tmp.right != nil; i++ {
		tmp = tmp.right
	}
	if tmp.left != nil {
		tmp = tmp.left
	}
	//随机从桶里面找到两个节点，并在此递归执行SetValue操作
	if len(tmp.kBucket) > 0 && tmp.kBucket[0] != nil {
		node1 := tmp.kBucket[0]
		node1.SetValue(key, value1)
	}
	if len(tmp.kBucket) > 1 && tmp.kBucket[1] != nil {
		node1 := tmp.kBucket[1]
		node1.SetValue(key, value1)
	}
	return true
}
//用于寻找当前key对应的value是多少
func (k *kBucketTreeNode) GetValue(key []byte) []byte {
	//如果当前节点找到了key，则直接返回对应的value
	if _, ok := k.value[string(key)]; ok {
		return k.value[string(key)]
	}
	k1 := kBucketTree{root: k}
	//否则找到两个最近的节点，这里节点数目不能确定，有可能改桶没有节点，故再次判断是否可以拿到一个节点或者两个节点，否则返回一个空
	tmpk := k1.FindNode(string(key))
	if len(tmpk) == 1 {
		if _, ok := tmpk[0].value[string(key)]; ok {
			return tmpk[0].value[string(key)]
		}
	} else if len(tmpk) == 2 {
		if _, ok := tmpk[0].value[string(key)]; ok {
			return tmpk[0].value[string(key)]
		}
		if _, ok := tmpk[1].value[string(key)]; ok {
			return tmpk[1].value[string(key)]
		}
	}
	return nil
}
//FindNode操作：根据当前的nodeId来进行查找对应的桶是否有这个节点，如果有返回这个节点，没有随便返回桶中的两个节点
func (k *kBucketTree) FindNode(nodeId string) []*kBucketTreeNode {
	belong := k.findKBucket([]byte(nodeId))
	tmp := k.root
	if belong != -1 {
		for i := 0; i < belong-1; i++ {
			tmp = tmp.right
		}
		if tmp.left != nil {
			tmp = tmp.left
		}
		for _, r := range tmp.kBucket {
			if bytes.Equal(r.nodeID, []byte(nodeId)) {
				return []*kBucketTreeNode{r}
			}
		}
		if len(tmp.kBucket) >= 2 {
			return []*kBucketTreeNode{tmp.kBucket[0], tmp.kBucket[1]}
		}
	}
	return nil
}

//测试
func main() {
	//随机生成100个[20]byte的节点id，并用hashmap记录唯一标识
	var byteSlices [100][20]byte
	byteMap := make(map[[20]byte]bool)
	for len(byteMap) < 100 {
		var byteSlice [20]byte
		_, err := rand.Read(byteSlice[:])
		if err != nil {
			panic(err)
		}
		if !byteMap[byteSlice] {
			byteMap[byteSlice] = true
			byteSlices[len(byteMap)-1] = byteSlice
		}
	}
	//初始化这个节点
	ansk :=make([]*kBucketTree,100)
	for i, r := range byteSlices {
		tmp:=make([]byte,len(r))
		copy(tmp,r[:])
		ansk[i] = InitkBucketTree(3, tmp)
	}
	//循环依次添加peer到peer中
	for _, r := range ansk {
		for _, r1 := range ansk {
			if r1!=r {
				r.addNode(r1.root)
			}
		}
	}
	//初始化字符串数组
	ansS := generateStr()
	//调用setvalue方法，把所有的key和对应的hash存入节点中
	for i, s := range ansS {
		new1 := hashnodeKey(s)
		new11 := new1[:]
		ansk[0].root.SetValue([]byte(ansS[i]), new11)
	}

	//随机选取一百个key，调用getvalue方法并打印对应的value
	for i := 0; i < 100; i++ {
		randnum := rand1.Intn(99) + 1
		zzzz := ansk[randnum].root.GetValue([]byte(ansS[randnum]))
		fmt.Println("key:",ansS[randnum],"value:",zzzz)
	}

}
//生成指定长度字符串
func stringWithCharset(length int, charset string) string {
	b := make([]byte, length)
	for i := range b {
		bigIndex, _ := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
		b[i] = charset[bigIndex.Int64()]
	}
	return string(b)
}
//连续生成两百个随机字符串
func generateStr() (ans []string) {
	strings := make(map[string]bool)
	count := 0
	// 生成直到生成100个独一无二的字符串
	for len(strings) < 200 {
		length, _ := rand.Int(rand.Reader, big.NewInt(256-strLen))
		strLength := int(length.Int64()) + strLen
		str := stringWithCharset(strLength, alphaNum)
		if !strings[str] {
			strings[str] = true
			count++
			ans = append(ans, str)
		}
	}
	return ans
}
