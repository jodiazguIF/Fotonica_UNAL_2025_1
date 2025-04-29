package Taller_1;

import java.util.List;
import java.util.ArrayList;

public class talller1_ResultadosOndulatorios {
    public static void main(String[] args) {
        double n_clad = 1.0; //Índice de refracción del revestimiento
        double n_core = 1.5; //Índice de refracción del núcleo 
        double long_Onda = 1E-6; //Longitud de onda
        double altura_fibra = 1E-6; //Altura de la fibra
        double numero_onda = (2*Math.PI)/long_Onda; //Número de onda    
        double fin = (numero_onda*altura_fibra)/2*Math.sqrt(Math.pow(n_core,2)-Math.pow(n_clad,2)); //Límite superior
        double inicio = 0; //Límite inferior
        // Modos Transversales Electricos Pares
        List <Double> raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTEPares, inicio, fin);
        List <Double> raices_alfa = new ArrayList<>();
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = phi * Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        List <Double> thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        System.out.println("Thetas Permitidos TE Pares: " + thetas_permitidos);
        raices_alfa.clear(); //Limpiamos la lista de raices_alfa

        // Modos Transversales Electricos Impares
        raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTEImpares, inicio, fin);
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = -phi / Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        System.out.println("Thetas Permitidos TE Impares: " + thetas_permitidos);
        raices_alfa.clear(); //Limpiamos la lista de raices_alfa

        // Modos Transversales Magneticos Pares
        raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTMPares, inicio, fin);
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = Math.pow(n_clad/n_core,2)*phi * Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        System.out.println("Thetas Permitidos TM Pares: " + thetas_permitidos);
        raices_alfa.clear(); //Limpiamos la lista de raices_alfa

        //Modos Transversales Magneticos Impares
        raices_phi = taller1_MetodoBiseccion.metodo_Biseccion(taller1_ModosOndulatorios::resultado_FuncionModosTMImpares, inicio, fin); 
        for (int i = 0; i < raices_phi.size(); i++){
            /*En este ciclo se calulan los valores de alfa */
            double phi = raices_phi.get(i); //Se obtiene el phi
            double alfa = -Math.pow(n_clad/n_core,2)*phi / Math.tan(phi); //Se obtiene el alfa
            raices_alfa.add(alfa); //Se agrega el alfa a la lista
        }
        thetas_permitidos = taller1_Pasos3_5MetodoOndulatorio.pasos_3_5MetodoOndulatorio(raices_phi, raices_alfa);
        System.out.println("Thetas Permitidos TM Impares: " + thetas_permitidos);
        
    }
}
