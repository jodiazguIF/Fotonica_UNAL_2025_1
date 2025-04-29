package Taller_1;

public class taller1_ModosOndulatorios {
    public static double resultado_FuncionModosTEPares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(phi*Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));
        /*Con esta ecuación es posible conocer los phi, con lo que es posible conocer los phi.
         * Esto ayuda a hallar los valores de kappa y gamma para finalmente obtener el beta.
         */
        return Ecuacion;
    }
    public static double resultado_FuncionModosTEImpares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(phi/Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));;
        return Ecuacion;
    }
    public static double resultado_FuncionModosTMPares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(Math.pow(n_clad/n_core,2)*phi*Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));
        return Ecuacion;
    }
    public static double resultado_FuncionModosTMImpares(double phi){
        double n_clad = 1.0 ; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double Ecuacion = Math.pow(phi,2)+Math.pow(Math.pow(n_clad/n_core,2)*phi/Math.tan(phi),2)
        -Math.pow((2*Math.PI*altura_fibra)/(long_Onda*2),2)*(Math.pow(n_core, 2)-Math.pow(n_clad, 2));
        return Ecuacion;
    }
}

