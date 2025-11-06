using FileIO
using DelimitedFiles
using Statistics
using Flux
using Flux.Losses
using Random
using Random:seed!
using PrettyTables
using RCall
using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface


SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0

Random.seed!(42)

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                                              CARGA DE DATOS
#---------------------------------------------------------------------------------------------------------------------------------------------

include("soluciones.jl")
dataset = readdlm("winequality-red.data", ';')

inputs = dataset[:, [2, 4, 5, 8, 10, 11]]
inputs = convert(Array{Float64}, inputs)
targets = dataset[:, 12]
targets_agrupados = map(x -> x ≤ 5 ? 0 : 1, targets)
#---------------------------------------------------------------------------------------------------------------------------------------------
#                                                              PREPROCESADO DE DATOS
#---------------------------------------------------------------------------------------------------------------------------------------------

k = 10
indices_cv = crossvalidation(targets_agrupados, k)

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                    CODIGO UTILIZADO PARA CREAR EL ARCHIVO DE INDICES ENTREGADO EN EL ZIP FINAL
#---------------------------------------------------------------------------------------------------------------------------------------------

# function save_all_indices(indices, filename)
#     open(filename, "w") do file
#         for fold in indices
#             for idx in fold
#                 println(file, idx)
#             end
#         end
#     end
#     println("Todos los índices guardados en $filename")
# end

# save_all_indices(indices_cv, "cv_indices.dat")

#---------------------------------------------------------------------------------------------------------------------------------------------

function print_results(config, results)
    # Define header and data for the pretty table
    header = ["Metric", "Average", "Std. Dev."]
    metrics = ["Precisión", "Tasa de error", "Sensibilidad", "Especificidad", "VPP", "VPN", "F1"]
    
    # Prepare data for the table
    data = Matrix{Any}(undef, length(metrics), 3)
    for i in 1:length(metrics)
        data[i, 1] = metrics[i]
        data[i, 2] = round(results[i][1], digits=3)
        data[i, 3] = round(results[i][2], digits=3)
    end
    
    # Print the table with formatting
    pretty_table(data, header=header, 
                 formatters=ft_printf("%.3f", [2, 3]),
                 alignment=[:l, :c, :c],
                 crop=:none)
    println()
    
end

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                                          ENTRENAMIENTO MEDIANTE RR.NN.AA
#---------------------------------------------------------------------------------------------------------------------------------------------

datos_normalizados = normalizeMinMax(inputs)

println("┌─────────────────────────────────────────────────────┐")
println("│              RESULTADOS REDES NEURONALES            │")
println("└─────────────────────────────────────────────────────┘")

topology = [
    [32, 16],
    [64, 32],
    [128, 64],
    [64, 16],
    [100, 50],
    [50, 25],
    [30, 10],
    [40, 20]
]

for (i, topo) in enumerate(topology)
    hyperparameters_nn = Dict("topology" => topo,
                             "maxEpochs" => 100,
                             "learningRate" => 0.01,
                             "numExecutions" => 5,
                             "validationRatio" => 0.2,
                             "maxEpochsVal" => 10)
    
    resultados_nn = modelCrossValidation(:ANN, hyperparameters_nn, (datos_normalizados, targets_agrupados), indices_cv)
    
    println("Topología: ", topo)
    print_results(hyperparameters_nn, resultados_nn)
    println(Int.(resultados_nn[8]))
end

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                                          ENTRENAMIENTO MEDIANTE SVM
#---------------------------------------------------------------------------------------------------------------------------------------------

datos_normalizados = normalizeZeroMean(inputs)

println("\n┌─────────────────────────────────────────────────────┐")
println("│                  RESULTADOS SVM                     │")
println("└─────────────────────────────────────────────────────┘")

configuraciones = [
    Dict("kernel" => "linear", "C" => 0.3),
    Dict("kernel" => "linear", "C" => 2.0),
    Dict("kernel" => "rbf", "C" => 0.3, "gamma" => 0.5),
    Dict("kernel" => "rbf", "C" => 2.0, "gamma" => 0.8),
    Dict("kernel" => "poly", "C" => 0.3, "degree" => 3, "gamma" => 0.5, "coef0" => 0),
    Dict("kernel" => "poly", "C" => 2.0, "degree" => 3, "gamma" => 0.8, "coef0" => 0),
    Dict("kernel" => "sigmoid", "C" => 0.1, "gamma" => 0.1, "coef0" => 0.0),
    Dict("kernel" => "sigmoid", "C" => 3.0, "gamma" => 0.1, "coef0" => 0.0)
]

for configuracion in configuraciones
    # Pretty print configuration
    println("Configuración: kernel=$(configuracion["kernel"]), C=$(configuracion["C"])" * 
            (haskey(configuracion, "gamma") ? ", gamma=$(configuracion["gamma"])" : "") *
            (haskey(configuracion, "degree") ? ", degree=$(configuracion["degree"])" : "") *
            (haskey(configuracion, "coef0") ? ", coef0=$(configuracion["coef0"])" : ""))
    
    resultados_svm = modelCrossValidation(:SVC, configuracion, (datos_normalizados, targets_agrupados), indices_cv)
    print_results(configuracion, resultados_svm)
    println(resultados_svm[8])
end

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                                          ENTRENAMIENTO MEDIANTE DoME
#---------------------------------------------------------------------------------------------------------------------------------------------

datos_normalizados = normalizeMinMax(inputs)

println("\n┌─────────────────────────────────────────────────────┐")
println("│                  RESULTADOS DoME                    │")
println("└─────────────────────────────────────────────────────┘")

valores_max_nodes = [7, 10, 15, 17, 19, 21, 23, 25]

for max_nodes in valores_max_nodes
    hyperparameters_DoME = Dict("maximumNodes" => max_nodes)
    resultados_DoME = modelCrossValidation(:DoME, hyperparameters_DoME, (datos_normalizados, targets_agrupados), indices_cv)
    
    println("Nodos máximos: $max_nodes")
    print_results(hyperparameters_DoME, resultados_DoME)
    println(resultados_DoME[8])
end

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                                  ENTRENAMIENTO MEDIANTE ÁRBOLES DE DECISIÓN
#---------------------------------------------------------------------------------------------------------------------------------------------

println("\n┌─────────────────────────────────────────────────────┐")
println("│            RESULTADOS ÁRBOLES DE DECISIÓN           │")
println("└─────────────────────────────────────────────────────┘")

valores_profundidad = [12, 14, 16, 19, 20, 23]

for profundidad in valores_profundidad
    hyperparameters_tree = Dict("max_depth" => profundidad)
    resultados_tree = modelCrossValidation(:DecisionTreeClassifier, hyperparameters_tree, (inputs, targets_agrupados), indices_cv)
    
    println("Profundidad: $profundidad")
    print_results(hyperparameters_tree, resultados_tree)
    println(resultados_tree[8])
end

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                                          ENTRENAMIENTO MEDIANTE kNN
#---------------------------------------------------------------------------------------------------------------------------------------------

datos_normalizados = normalizeMinMax(inputs)

println("\n┌─────────────────────────────────────────────────────┐")
println("│                  RESULTADOS kNN                     │")
println("└─────────────────────────────────────────────────────┘")

valores_k = [9, 11, 13, 15, 17, 19]

for k_value in valores_k
    hyperparameters_knn = Dict("n_neighbors" => k_value)
    resultados_knn = modelCrossValidation(:KNeighborsClassifier, hyperparameters_knn, (datos_normalizados, targets_agrupados), indices_cv)
    
    println("k = $k_value")
    print_results(hyperparameters_knn, resultados_knn)
    println(resultados_knn[8])
end

#---------------------------------------------------------------------------------------------------------------------------------------------
#                                         CODIGO R CON EL QUE GENERAMOS LAS GRAFICAS MOSTRADAS EN EL PDF
#---------------------------------------------------------------------------------------------------------------------------------------------

R"""
source('vino.R')
"""