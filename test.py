def solution(L, x):
    i = 0
    lower = 0
    upper = len(L)
    answer = 0
    while True:
        middle = round((lower+upper) / 2)

        # L이 없는 경우; UPPER 또는 LOWER 포인터 이동
        try: 
            if(L[middle] == x) :
                answer = middle
                break
            elif(i == middle) :
                answer = -1
                break
            elif(L[middle] < x) :
                lower = middle+1
            elif(L[middle] > x) :
                upper = middle-1

            i=middle
        except IndexError :
            answer = -1
            break
    print(answer)
    return answer

L = [20, 37, 58, 72, 91, 5, 8]
x = 100
solution(L, x)

def solution(L, x):
    # index 정의
    start, end = 0, len(L)-1

    while start <= end:
        mid = (start+end)//2
        if L[mid] == x:
            return mid
        elif L[mid] > x:
            end = mid-1
        elif L[mid] < x: 
            start = mid+1

    return -1