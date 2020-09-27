import java.util.Random; //가우시안 함수를 사용하기 위해 불러온다.

public class Backpropagation {
	
	static final double e = 2.71828182846; //e 정의
	static final double eta = 0.1; //eta = 0.1로 정의
	static final int IN = 25; //입력층 노드 개수
	static final int HL = 20; //은닉층 노드 개수
	static final int OUT = 12; //출력층 노드 개수
	static final double delta_range = 0.00001; //가중치 차이(오차) 범위
	
	static double wIH[][] = new double[HL][IN];
	static double wHO[][] = new double[OUT][HL];
	static double wIH_copy[][] = new double[HL][IN];
	static double wHO_copy[][] = new double[OUT][HL];
	
	static double HiddenLayer[] = new double[HL];
	static double OutputLayer[] = new double[OUT];
	static double Hiddendelta[] = new double[HL];
	static double Outputdelta[] = new double[OUT];
	
	//학습 패턴
	private static final int PATTERN[][] = {
			{	//'ㄱ' 패턴
				1,1,1,1,1,       
				1,1,1,1,1,
				0,0,0,1,1,
				0,0,0,1,1,
				0,0,0,1,1	},

			{	//'ㄴ' 패턴
				1,1,0,0,0,
				1,1,0,0,0,
				1,1,0,0,0,
				1,1,1,1,1,
				1,1,1,1,1	},

			{	//'ㄷ' 패턴
				1,1,1,1,1,
				1,1,1,1,1,
				1,1,0,0,0,
				1,1,1,1,1,
				1,1,1,1,1	},

			{	//'ㄹ' 패턴
				1,1,1,1,1,
				0,0,0,0,1,
				1,1,1,1,1,
				1,0,0,0,0,
				1,1,1,1,1	},

			{	//'ㅁ' 패턴
				1,1,1,1,1,
				1,1,1,1,1,
				1,1,0,1,1,
				1,1,1,1,1,
				1,1,1,1,1	},
			
			{	//'ㅂ' 패턴
				1,1,0,1,1,
				1,1,0,1,1,
				1,1,1,1,1,
				1,1,0,1,1,
				1,1,1,1,1	},

			{	//'ㅅ' 패턴
				0,0,1,0,0,
				0,0,1,0,0,
				0,1,1,1,0,
				1,1,0,1,1,
				1,0,0,0,1	},

			{	//'ㅇ' 패턴
				0,1,1,1,0,
				1,1,1,1,1,
				1,1,0,1,1,
				1,1,0,1,1,
				0,1,1,1,0	},
			
			{	//'ㅈ' 패턴
				1,1,1,1,1,
				0,0,1,0,0,
				0,1,1,1,0,
				1,1,0,1,1,
				1,0,0,0,1	},
			
			{	//'ㅊ' 패턴
				0,0,1,0,0,
				1,1,1,1,1,
				0,0,1,0,0,
				1,1,0,1,1,
				1,0,0,0,1	},
			
			{	//'ㅋ' 패턴
				1,1,1,1,1,
				0,0,0,0,1,
				1,1,1,1,1,
				0,0,0,0,1,
				0,0,0,0,1	},
				
			{	//'ㅌ' 패턴
				1,1,1,1,1,
				1,0,0,0,0,
				1,1,1,1,1,
				1,0,0,0,0,
				1,1,1,1,1	}
			
		};	   
	
	//노이즈 패턴
	private static final int PATTERNTEST[][] = {
			{	//'ㄱ' 노이즈 패턴 
				0,1,1,1,1,       
				0,1,1,1,1,
				0,0,0,1,1,
				0,0,0,1,1,
				0,0,0,0,0	},

			{	//'ㄴ' 노이즈 패턴
				0,0,0,0,0,
				1,1,0,0,0,
				1,1,0,0,0,
				1,1,1,1,0,
				1,1,1,1,0	},

			{	//'ㄷ' 노이즈 패턴
				1,1,1,1,0,
				1,1,1,1,0,
				1,1,0,0,0,
				1,1,1,1,0,
				1,1,1,1,0	},

			{	//'ㄹ' 노이즈 패턴
				0,1,1,1,1,
				0,0,0,0,1,
				0,1,1,1,0,
				1,0,0,0,0,
				1,1,1,1,0	},

			{	//'ㅁ' 노이즈 패턴
				1,1,1,1,1,
				1,1,1,1,1,
				1,0,0,0,1,
				1,1,1,1,1,
				1,1,1,1,1	},
			
			{	//'ㅂ' 노이즈 패턴
				0,0,0,0,0,
				1,1,0,1,1,
				1,1,1,1,1,
				1,1,0,1,1,
				1,1,1,1,1	},

			{	//'ㅅ' 노이즈 패턴
				0,0,0,0,0,
				0,0,1,0,0,
				0,1,0,1,0,
				1,0,0,0,1,
				0,0,0,0,0	},

			{	//'ㅇ' 노이즈 패턴
				0,1,1,1,0,
				1,0,0,0,1,
				1,0,0,0,1,
				1,0,0,0,1,
				0,1,1,1,0	},
			
			{	//'ㅈ' 노이즈 패턴
				1,1,1,1,1,
				0,0,1,0,0,
				0,1,1,1,0,
				1,1,1,1,1,
				1,0,1,0,1	},
				
			{	//'ㅊ' 노이즈 패턴
				0,0,1,0,0,
				1,1,1,1,1,
				0,0,1,0,0,
				1,1,1,1,1,
				1,1,1,1,1	},
			
			{	//'ㅋ' 노이즈 패턴
				1,1,1,1,1,
				0,0,0,0,1,
				0,1,1,1,1,
				0,0,0,0,1,
				0,0,1,1,1	},
					
			{	//'ㅌ' 노이즈 패턴
				1,1,1,1,1,
				1,1,1,0,0,
				1,1,1,1,1,
				1,1,1,0,0,
				1,1,1,1,1	}
		};	   
	   
	private static double sigmoid(double x) { //시그모이드 함수
		return 1 / (1 + Math.pow(e, -x));
	}
	
	// 분포가 가운데에서 종 모양을 이루는 가우시안 랜덤 실수로 가중치 초기화
    // 평균은 0.0이고, 표준편차는 1.0인 가우시안 분포에 따른 double형의 난수 생성
	public static void Rand_wIH(){ //입력층->은닉층 가중치 초기화
		Random random = new Random();
		for (int i=0;i<HL;i++) {
		      for(int j=0;j<IN;j++) {
		    	  wIH[i][j]=random.nextGaussian()/Math.sqrt(IN);}
		      }
		}
	
	public static void Rand_wHO() { //은닉층->출력층 가중치 초기화
		Random random = new Random();
		for (int i=0;i<OUT;i++) {
		      for(int j=0;j<HL;j++) {
		    	  wHO[i][j]=random.nextGaussian()/Math.sqrt(IN);}
		      }
		}
	
	//입력층->은닉층의 출력값
	public static void Calc_HL(int PATTERN[]) {
		double sum[]=new double[HL];
		
		for(int i=0;i<HL;i++) {
			for(int j=0;j<IN;j++) {
				sum[i] += PATTERN[j] * wIH[i][j];
			}
			HiddenLayer[i] = sigmoid(sum[i]);
		}
	}
	
	//은닉층->출력층의 출력값
	public static void Calc_OUT() {
		double sum[]=new double[OUT];
		
		for(int i=0;i<OUT;i++) {
			for(int j=0;j<HL;j++) {
				sum[i] += HiddenLayer[j] * wHO[i][j];
			}
			OutputLayer[i] = sigmoid(sum[i]);
		}
	}
	
	//출력층 오차 계산
	public static void Calc_Outputdelta(int index) {
		int output = 0;
		for(int i=0;i<OUT;i++) {
			if(index==i) {
				output = 1;
			}
			else {
				output = 0;
			}
			Outputdelta[i] = OutputLayer[i] * (1 - OutputLayer[i])*(output - OutputLayer[i]);
		}
	}
	
	//은닉층 오차 계산
	public static void Calc_Hiddendelta() {
		double sum = 0;
		for(int i=0;i<HL;i++) {
			sum = 0;
			for(int j=0;j<OUT;j++) {
				sum += Outputdelta[j]* wHO[j][i];
			}
			Hiddendelta[i] = HiddenLayer[i] * (1 - HiddenLayer[i]) * sum;
		}
	}
	
	//가중치 수정(은닉층->출력층)
		public static void Update_wHO() {
			for(int i=0;i<HL;i++) {
				for(int j=0; j<OUT; j++) {
					wHO[j][i] += eta * Outputdelta[j] * HiddenLayer[i]; 
					}
				}
			}
	
	//가중치 수정(입력층->은닉층)
	public static void Update_wIH(int input[]) {
		for(int i=0;i<IN;i++) {
			for(int j=0; j<HL; j++) {
				wIH[j][i] += eta * Hiddendelta[j] * input[i]; 
				}
			}
		}
			
	//입력층->은닉층 가중치 복사
	public static void Copy_wIH() {
		for (int i = 0; i < HL; i++) {
			for (int j = 0; j < IN; j++) {
				wIH_copy[i][j] = wIH[i][j];
			}
		}
	}
	
	//은닉층->출력층 가중치 복사
	public static void Copy_wHO() {
		for (int i = 0; i < OUT; i++) {
			for (int j = 0; j < HL; j++) {
				wHO_copy[i][j] = wHO[i][j];
			}
		}
	}
	
	public static boolean Escape_loop()
	{
		double delta = 0;
		for (int i = 0; i < HL; i++) {
			for (int j = 0; j < IN; j++) {
				delta = wIH[i][j] - wIH_copy[i][j];
				if (delta > delta_range || delta < -delta_range) {
					Copy_wIH();
					Copy_wHO();
					return false;
				}
			}
		}
		for (int i = 0; i < OUT; i++) {
			for (int j = 0; j < HL; j++) {
				delta = wHO[i][j] - wHO_copy[i][j];
				if (delta > delta_range || delta < -delta_range) {
					Copy_wIH();
					Copy_wHO();
					return false;
				}
			}
		}
		return true;
	}
	
	public static void Print_PATTERN(int PATTERN[][]) {
		for(int i=0;i<OUT;i++) {
			for(int j=0;j<IN;j++) {
				if(j!=0 && j%5==0) {
					System.out.println("");
				}
				if(PATTERN[i][j]==1) {
					System.out.print("■ ");
				}
				else {
					System.out.print("□ ");
				}
			}
			System.out.println("\n-----------------------");
		}
	}
	
	public static void compute(){ //가중치 오차 줄인 후 출력값 구하는 함수		
		double max=OutputLayer[0];
		
		for(int i=0;i<OUT;i++) {
			if(max<OutputLayer[i]) {
				max=OutputLayer[i];
			}
		}
		
		for(int i=0;i<OUT;i++) {
			System.out.print((int)(OutputLayer[i]/max)+"  ");
		}
		System.out.println("");		
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
	    Rand_wIH(); //가중치 초기화
	    Rand_wHO(); //가중치 초기화
	    
		boolean escape = false;
		
		int epoch =0;
		
		long start = System.currentTimeMillis();
		
		while(!escape) {
			for(int i=0;i<OUT;i++) {
				Calc_HL(PATTERN[i]);
				Calc_OUT();
				Calc_Outputdelta(i);
				Calc_Hiddendelta();
				Update_wHO();
				Update_wIH(PATTERN[i]);
			}
			epoch++;
			escape = Escape_loop();
		}
		long end = System.currentTimeMillis();
		
		for(int i=0;i<OUT;i++) {
			for(int j=0;j<HL;j++) {
				HiddenLayer[j]=0;
			}
			OutputLayer[i]=0;
		}
		
		System.out.println("▶▶학습할 패턴◀◀\n");
		Print_PATTERN(PATTERN);
		
		System.out.println("▶▶노이즈 패턴◀◀\n");
		Print_PATTERN(PATTERNTEST);
		
		System.out.println("\n↓↓↓↓↓↓↓↓↓↓");
		System.out.println("↓↓↓↓↓↓↓↓↓↓\n");
		System.out.println("▶▶학습 결과◀◀\n");
		System.out.println("epoch : "+epoch+"\n");
		System.out.println("학습 시간 : " + (end - start)/1000.0+"초\n");
		for(int i=0;i<OUT;i++) {
			Calc_HL(PATTERNTEST[i]);
			Calc_OUT();
			compute();
			}
		
	}
}

