using Statistics
using Flux
using Flux.Losses
using Random

# EJERCICIO 2 #

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if length(classes) <= 2
        # Si hay 2 o menos clases, se devuelve un array de una sola columna con True/False para la primera clase
        return reshape(feature .== classes[1], :, 1)
    else
        # Si hay más de 2 clases, se genera una matriz donde cada columna representa la pertenencia a una clase
        return hcat([feature .== c for c in classes]...)
    end
end

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)  # Obtiene las clases únicas en la característica
    return oneHotEncoding(feature, classes)  # Llama a la función principal con las clases identificadas
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1)  # Asegura que la salida tenga una dimensión adecuada
end

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    min_vals = minimum(dataset, dims=1)  # Obtiene los valores mínimos de cada columna
    max_vals = maximum(dataset, dims=1)  # Obtiene los valores máximos de cada columna
    return min_vals, max_vals  # Devuelve los valores mínimos y máximos para normalizar los datos
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mean_vals = mean(dataset, dims=1)  # Calcula la media de cada columna
    std_vals = std(dataset, dims=1)  # Calcula la desviación estándar de cada columna
    return mean_vals, std_vals  # Devuelve la media y la desviación estándar para normalizar los datos
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    min_vals, max_vals = normalizationParameters  # Extrae los valores mínimo y máximo por columna
    dataset .= (dataset .- min_vals) ./ (max_vals .- min_vals)  # Aplica la normalización Min-Max elemento a elemento
    dataset[:, vec(min_vals .== max_vals)] .= 0  # Si min y max son iguales en una columna, establece los valores en 0
    return dataset  # Devuelve el dataset normalizado
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    min_vals, max_vals = normalizationParameters  # Extrae los valores mínimo y máximo por columna
    normalized_dataset = (dataset .- min_vals) ./ (max_vals .- min_vals)  # Aplica la normalización Min-Max
    normalized_dataset[:, vec(min_vals .== max_vals)] .= 0  # Si min y max son iguales, establece los valores en 0
    return normalized_dataset  # Devuelve el dataset normalizado sin modificar el original
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    return normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset))
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean_vals, std_vals = normalizationParameters  # Extrae la media y la desviación estándar por columna
    dataset .= (dataset .- mean_vals) ./ (std_vals)  # Aplica la normalización elemento a elemento
    dataset[:, vec(std_vals .== 0)] .= 0  # Si la desviación estándar es 0, establece los valores en 0
    return dataset  # Devuelve el dataset normalizado
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset))
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean_vals, std_vals = normalizationParameters  # Extrae la media y la desviación estándar por columna
    normalized_dataset = (dataset .- mean_vals) ./ std_vals  # Aplica la normalización
    normalized_dataset[:, vec(std_vals .== 0)] .= 0  # Si la desviación estándar es 0, establece los valores en 0
    return normalized_dataset  # Devuelve el dataset normalizado sin modificar el original
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    return normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset))
end

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold  # Devuelve un vector booleano basado en el umbral
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        # Si solo hay una columna (caso binario), clasifica cada valor según el umbral
        return reshape(classifyOutputs(outputs[:,1], threshold=threshold), :, 1)
    else
        # Caso multiclase: Se obtiene el índice de la clase con mayor probabilidad en cada fila
        max_indices = getindex.(argmax(outputs, dims=2), 2)

        # Generar la matriz one-hot sin usar bucles explícitos
        oneHot = falses(size(outputs))  # Matriz de valores `false` del mismo tamaño que `outputs`
        oneHot[CartesianIndex.(1:size(outputs, 1), max_indices)] .= true  # Se activa la clase predicha en cada fila

        return oneHot  # Devuelve la matriz en formato one-hot
    end
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)  # Devuelve la precisión promedio
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1
        # Si es un problema binario con una sola columna, llamar a la función de vectores
        return accuracy(outputs[:,1], targets[:,1])
    else
        # En el caso multiclase, verifica si todas las columnas de cada fila coinciden con los targets
        return mean(all(outputs .== targets, dims=2))  # Calcula la precisión promedio por fila
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs_bool = classifyOutputs(outputs, threshold=threshold)  # Convierte las salidas en booleanas
    return accuracy(outputs_bool, targets)  # Llama a la función de precisión para vectores booleanos
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        # Si es un problema binario con una sola columna, umbralizar y calcular precisión
        return accuracy(outputs[:,1], targets[:,1], threshold=threshold)
    else
        # En el caso multiclase, clasificar las salidas y compararlas con los targets
        return accuracy(classifyOutputs(outputs), targets)
    end
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann = Chain()  # Inicializa una cadena vacía para la red neuronal
    numInputsLayer = numInputs  # El número de neuronas en la capa de entrada es igual al número de entradas

    # Construye las capas ocultas de la red neuronal según la topología proporcionada
    if !isempty(topology)
        for (i, numOutputsLayer) in enumerate(topology)
            ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))  # Añade una capa densa con función de activación
            numInputsLayer = numOutputsLayer  # Actualiza el número de entradas para la siguiente capa
        end
    end

    # Capa de salida
    if numOutputs == 1
        # Si hay una sola salida, utiliza una función de activación sigmoide
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, σ))
    else
        # Si hay más de una salida, usa la identidad como función de activación
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        
        # Si hay más de dos salidas, aplica la función softmax (usada en clasificación multiclase)
        if numOutputs > 2
            ann = Chain(ann..., softmax)
        end
    end

    return ann  # Devuelve la red neuronal construida
end

# EJERCICIO 3 #

function holdOut(N::Int, P::Real)
    indices = randperm(N)  # Genera una permutación aleatoria de los índices
    tests = ceil(Int, N * P)  # Calcula el número de ejemplos para el conjunto de test
    indices_entrenamiento = indices[1:N-tests]  # Los primeros N-tests índices son para el conjunto de entrenamiento
    indices_test = indices[N-tests+1:end]  # El resto de los índices son para el conjunto de test
    return (indices_entrenamiento, indices_test)  # Devuelve los índices de entrenamiento y test
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    _, indices_test = holdOut(N, Ptest)  # Primero se divide en entrenamiento y test usando la función anterior
    N_restante = N - length(indices_test)  # Se ajusta el valor de N_restante para reflejar los datos restantes después de separar el conjunto de test
    Pval_adjusted = Pval * N / N_restante  # Se ajusta el porcentaje de validación basado en el tamaño restante de los datos
    indices_entrenamiento, indices_validacion = holdOut(N_restante, Pval_adjusted)  # Luego se divide el conjunto restante en entrenamiento y validación
    return (indices_entrenamiento, indices_validacion, indices_test)  # Devuelve los índices de entrenamiento, validación y test
end

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    (trainingInputs, trainingTargets) = trainingDataset  # Separa las entradas y salidas del conjunto de entrenamiento
    (validationInputs, validationTargets) = validationDataset  # Separa las entradas y salidas del conjunto de validación
    (testInputs, testTargets) = testDataset  # Separa las entradas y salidas del conjunto de test

    # Construye el modelo de la red neuronal utilizando la función buildClassANN
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions)

    # Definimos la funcion de loss
    loss(model,x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);

    # Inicializa vectores para almacenar las pérdidas de entrenamiento, validación y test
    trainingLosses, validationLosses, testLosses = Float32[], Float32[], Float32[]  
    Epoch = 0  # Inicializa el contador de épocas

    # Función para calcular la pérdida de entrenamiento, validación y test
    function calculateLoss()
        trainingLoss = loss(ann, trainingInputs', trainingTargets')  # Calcula la pérdida de entrenamiento
        push!(trainingLosses, trainingLoss)

        validationLoss = NaN  # Inicializa la pérdida de validación
        if !isempty(validationInputs)  # Si hay datos de validación
            validationLoss = loss(ann, validationInputs', validationTargets')  # Calcula la pérdida de validación
            push!(validationLosses, validationLoss)
        end

        testLoss = NaN  # Inicializa la pérdida de test
        if !isempty(testInputs)  # Si hay datos de test
            testLoss = loss(ann, testInputs', testTargets')  # Calcula la pérdida de test
            push!(testLosses, testLoss)
        end

        return trainingLoss, validationLoss, testLoss  # Devuelve las pérdidas
    end

    # Calcula las pérdidas iniciales
    trainingLoss, validationLoss, _ = calculateLoss()

    # Si no hay datos de validación, no se limita el número de épocas de validación
    if isempty(validationInputs) 
        maxEpochsVal = Inf 
    end

    # Inicializa variables para el control de las mejores pérdidas durante el entrenamiento
    EpochsValidation, bestValidationLoss = 0, validationLoss
    bestANN = deepcopy(ann)  # Guarda una copia del mejor modelo
    opt_state = Flux.setup(Adam(learningRate), ann)  # Configura el optimizador (Adam)

    # Ciclo de entrenamiento
    while Epoch < maxEpochs && trainingLoss > minLoss && EpochsValidation < maxEpochsVal
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt_state)  # Entrenamiento de una época
        Epoch += 1  # Incrementa el contador de épocas
        trainingLoss, validationLoss, _ = calculateLoss()  # Calcula las pérdidas después de la época

        # Si hay datos de validación
        if !isempty(validationInputs)
            if validationLoss < bestValidationLoss  # Si la pérdida de validación ha mejorado
                EpochsValidation, bestValidationLoss = 0, validationLoss  # Reinicia el contador de épocas sin mejora
                bestANN = deepcopy(ann)  # Guarda el modelo actual como el mejor
            else
                EpochsValidation += 1  # Si no mejora, aumenta el contador de épocas sin mejora
            end
        end
    end

    # Si no hay validación, el mejor modelo es el modelo final
    if isempty(validationInputs)
        bestANN = ann
    end

    return bestANN, trainingLosses, validationLosses, testLosses  # Devuelve el mejor modelo y las pérdidas
end


function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    (trainingInputs, trainingTargets) = trainingDataset  # Separa las entradas y salidas del conjunto de entrenamiento
    (validationInputs, validationTargets) = validationDataset  # Separa las entradas y salidas del conjunto de validación
    (testInputs, testTargets) = testDataset  # Separa las entradas y salidas del conjunto de test

    # Llama a la primera función, pero adaptando los targets a un formato adecuado
    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1));
        validationDataset=(validationInputs, reshape(validationTargets, length(validationTargets), 1)), 
        testDataset=(testInputs, reshape(testTargets, length(testTargets), 1)), 
        transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
end

# EJERCICIO 4 #

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    VP = sum(outputs .& targets)

    precision = (VN + VP) / (VN + VP + FN + FP)

    tasa_error = (FN + FP) / (VN + VP + FN + FP)
    
    if VP == FN == 0
        sensibilidad = 1
    else
        sensibilidad = VP / (FN + VP)
    end
    
    if VN == FP == 0
        especificidad = 1
    else
        especificidad = VN / (FP + VN)
    end

    if VP == FP == 0
        VPP = 1
    else
        VPP = VP / (VP + FP)
    end
    
    if VN == FN == 0
        VPN = 1
    else
        VPN = VN / (VN + FN)
    end

    if sensibilidad == VPP == 0
        F1 = 0
    else
        F1 = 2 * (sensibilidad * VPP) / (sensibilidad + VPP)
    end

    matriz_confusion = [VN FP; FN VP]
    return (precision, tasa_error, sensibilidad, especificidad, VPP, VPN, F1, matriz_confusion)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return confusionMatrix(outputs .>= threshold, targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs) == size(targets))
    numClasses = size(targets, 2)
    @assert(numClasses!=2)
    
    if numClasses == 1
        return confusionMatrix(outputs[:,1], targets[:,1])
    end

    @assert(all(sum(outputs, dims = 2) .== 1))

    sensibilidad = zeros(numClasses)
    especificidad = zeros(numClasses)
    VPP = zeros(numClasses)
    VPN = zeros(numClasses)
    F1 = zeros(numClasses)

    for numClass in 1:numClasses
        (_, _, sensibilidad[numClass], especificidad[numClass], VPP[numClass], VPN[numClass], F1[numClass], _) = confusionMatrix(outputs[:, numClass], targets[:, numClass])
    end

    matriz_confusion = zeros(Int, numClasses, numClasses)   
    for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
        matriz_confusion[numClassTarget, numClassOutput] = sum(targets[:, numClassTarget] .& outputs[:, numClassOutput])
    end

    if weighted
        numInstancesFromEachClass = vec(sum(targets, dims=1))
        weights = numInstancesFromEachClass ./ sum(numInstancesFromEachClass)
        sensibilidad = sum(weights .* sensibilidad)
        especificidad = sum(weights .* especificidad)
        VPP = sum(weights .* VPP)
        VPN = sum(weights .* VPN)
        F1 = sum(weights .* F1)
    else
        sensibilidad = mean(sensibilidad)
        especificidad = mean(especificidad)
        VPP = mean(VPP)
        VPN = mean(VPN)
        F1 = mean(F1)
    end

    precision = accuracy(outputs, targets)
    tasa_error = 1 - precision

    return (precision, tasa_error, sensibilidad, especificidad, VPP, VPN, F1, matriz_confusion)
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    return confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]))
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end

using SymDoME

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = Float64.(trainingDataset[1])
    trainingTargets = trainingDataset[2]
    testInputs = Float64.(testInputs)
    
    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes)

    testOutputs = evaluateTree(model, testInputs)
    
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1))
    end

    return testOutputs
end

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = Float64.(trainingDataset[1])
    trainingTargets = trainingDataset[2]
    testInputs = Float64.(testInputs)

    if size(trainingTargets, 2) == 1
        return reshape(trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes), :, 1)
    else
        numClasses = size(trainingTargets, 2)
        testOutputs = zeros(Float64, size(testInputs, 1), numClasses)

        for classIndex in 1:numClasses
            columnTargets = vec(trainingTargets[:, classIndex])
            testOutputs[:, classIndex] = reshape(trainClassDoME((trainingInputs, columnTargets), testInputs, maximumNodes), :, 1)
        end

        return testOutputs
    end
end


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = Float64.(trainingDataset[1])
    trainingTargets = trainingDataset[2]
    testInputs = Float64.(testInputs)
    
    classes = unique(trainingTargets)
    
    testOutputs = Array{eltype(trainingTargets), 1}(undef, size(testInputs, 1))
    
    testOutputsDoME = trainClassDoME(
        (trainingInputs, oneHotEncoding(trainingTargets, classes)),  
        testInputs, maximumNodes)
    
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0)
    
    if length(classes) <= 2
        testOutputsBool = vec(testOutputsBool)
        testOutputs[testOutputsBool] .= classes[1]
        if length(classes) == 2
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    else

        for numClass in 1:length(classes)
            testOutputs[testOutputsBool[:, numClass]] .= classes[numClass]
        end
    end
    
    return testOutputs
end

# EJERCICIO 5 #

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)))
    indices = indices[1:N]
    shuffle!(indices)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = Array{Int64,1}(undef, length(targets))
    num_positivos = sum(targets)
    num_negativos = sum(.!targets)
    indices[targets] .= crossvalidation(num_positivos, k)
    indices[.!targets] .= crossvalidation(num_negativos, k)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    numClasses = size(targets, 2)
    indices = Array{Int64,1}(undef, size(targets, 1))
    for numClass in 1:numClasses
        indices_class = targets[:, numClass] 
        indices[indices_class] = crossvalidation(sum(targets[:, numClass]), k)
    end    
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    return crossvalidation(oneHotEncoding(targets), k)
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    
    inputs, targets = dataset
    classes = unique(targets)
    targets = oneHotEncoding(targets, classes)

    folds = maximum(crossValidationIndices)
    precision = zeros(Float64, folds)
    tasa_error = zeros(Float64, folds)
    sensibilidad = zeros(Float64, folds)
    especificidad = zeros(Float64, folds)
    VPP = zeros(Float64, folds)
    VPN = zeros(Float64, folds)
    F1 = zeros(Float64, folds)
    numClasses = length(classes)
    matriz_confusion = zeros(Int, numClasses, numClasses)

    for fold in 1:folds
        
        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        testInputs = inputs[crossValidationIndices .== fold, :]
        trainingTargets = targets[crossValidationIndices .!= fold, :]
        testTargets = targets[crossValidationIndices .== fold, :]

        precision_fold = zeros(Float64, numExecutions)
        tasa_error_fold = zeros(Float64, numExecutions)
        sensibilidad_fold = zeros(Float64, numExecutions)
        especificidad_fold = zeros(Float64, numExecutions)
        VPP_fold = zeros(Float64, numExecutions)
        VPN_fold = zeros(Float64, numExecutions)
        F1_fold = zeros(Float64, numExecutions)
        matriz_confusion_fold = zeros(Int, numClasses, numClasses, numExecutions)

        for train in 1:numExecutions
            
            if validationRatio > 0

                (trainingIndices, validationIndices) = holdOut(size(trainingInputs, 1), validationRatio * size(inputs, 1) / size(trainingInputs, 1))

                ann, _, _, _ = trainClassANN(topology, (trainingInputs[trainingIndices, :], trainingTargets[trainingIndices, :]),
                    validationDataset=(trainingInputs[validationIndices, :], trainingTargets[validationIndices, :]),
                    testDataset=(testInputs, testTargets);
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
 
            else

                ann, _, _, _ = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs, testTargets);
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);

            end

            (precision_fold[train], tasa_error_fold[train], sensibilidad_fold[train], especificidad_fold[train], VPP_fold[train], VPN_fold[train], F1_fold[train], matriz_confusion_fold[:, :, train]) = confusionMatrix(collect(ann(testInputs')'), testTargets)
 
        end

        precision[fold] = mean(precision_fold)
        tasa_error[fold] = mean(tasa_error_fold)
        sensibilidad[fold] = mean(sensibilidad_fold)
        especificidad[fold] = mean(especificidad_fold)
        VPP[fold] = mean(VPP_fold)
        VPN[fold] = mean(VPN_fold)
        F1[fold] = mean(F1_fold)
        matriz_confusion_fold_promedio = mean(matriz_confusion_fold, dims=3)[:, :, 1]
        matriz_confusion += matriz_confusion_fold_promedio

    end

    precision_result = (mean(precision), std(precision))
    tasa_error_result = (mean(tasa_error), std(tasa_error))
    sensibilidad_result = (mean(sensibilidad), std(sensibilidad))
    especificidad_result = (mean(especificidad), std(especificidad))
    VPP_result = (mean(VPP), std(VPP))
    VPN_result = (mean(VPN), std(VPN))
    F1_result = (mean(F1), std(F1))

    return precision_result, tasa_error_result, sensibilidad_result, especificidad_result, VPP_result, VPN_result, F1_result, matriz_confusion

end

# EJERCICIO 6 #

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    
    inputs, targets = dataset
    targets = string.(targets)
    classes = unique(targets)
    
    if modelType==:ANN

       return ANNCrossValidation(get(modelHyperparameters, "topology", [10,5]),
            (inputs, targets), crossValidationIndices,
            numExecutions     = get(modelHyperparameters, "numExecutions",     50),
            transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(modelHyperparameters["topology"]))),
            maxEpochs         = get(modelHyperparameters, "maxEpochs",         1000),
            minLoss           = get(modelHyperparameters, "minLoss",           0.0),
            learningRate      = get(modelHyperparameters, "learningRate",      0.01),
            validationRatio   = get(modelHyperparameters, "validationRatio",   0),
            maxEpochsVal      = get(modelHyperparameters, "maxEpochsVal",      20))

    end
    
    folds = maximum(crossValidationIndices)
    precision = zeros(Float64, folds)
    tasa_error = zeros(Float64, folds)
    sensibilidad = zeros(Float64, folds)
    especificidad = zeros(Float64, folds)
    VPP = zeros(Float64, folds)
    VPN = zeros(Float64, folds)
    F1 = zeros(Float64, folds)
    numClasses = length(classes)
    matriz_confusion = zeros(Int, numClasses, numClasses, folds)

    for fold in 1:folds

        trainingInputs = inputs[crossValidationIndices .!= fold, :]
        testInputs = inputs[crossValidationIndices .== fold, :]
        trainingTargets = targets[crossValidationIndices .!= fold]
        testTargets = targets[crossValidationIndices .== fold]

        if modelType == :DoME
            testOutputs = trainClassDoME((trainingInputs, trainingTargets), testInputs, modelHyperparameters["maximumNodes"] )

        elseif modelType == :SVC
            kernelDict = Dict("linear" => LIBSVM.Kernel.Linear, "rbf" => LIBSVM.Kernel.RadialBasis, "sigmoid" => LIBSVM.Kernel.Sigmoid, "poly" => LIBSVM.Kernel.Polynomial)
            kernelNew = kernelDict[modelHyperparameters["kernel"]]
            model = SVMClassifier(kernel = kernelNew, cost = Float64(modelHyperparameters["C"]))

            if modelHyperparameters["kernel"] == "poly"
                model.degree = Int32(modelHyperparameters["degree"])
                model.gamma = Float64(modelHyperparameters["gamma"])
                model.coef0 = Float64(modelHyperparameters["coef0"])

            elseif modelHyperparameters["kernel"] in ["rbf", "sigmoid"]
                model.gamma = Float64(modelHyperparameters["gamma"])

                if modelHyperparameters["kernel"] == "sigmoid"
                    model.coef0 = Float64(modelHyperparameters["coef0"])
                end
            end

            mach = machine(model, MLJ.table(trainingInputs), categorical(trainingTargets))
            MLJ.fit!(mach, verbosity=0)
            testOutputs = MLJ.predict(mach, MLJ.table(testInputs))

        elseif modelType==:DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"], rng=1)

            mach = machine(model, MLJ.table(trainingInputs), categorical(trainingTargets))
            MLJ.fit!(mach, verbosity=0)
            testOutputs = mode.(MLJ.predict(mach, MLJ.table(testInputs)))

        elseif modelType==:KNeighborsClassifier
            model = KNNClassifier(K = modelHyperparameters["n_neighbors"])

            mach = machine(model, MLJ.table(trainingInputs), categorical(trainingTargets))
            MLJ.fit!(mach, verbosity=0)
            testOutputs = mode.(MLJ.predict(mach, MLJ.table(testInputs)))
        
        else
            error(string("Unknown model ", modelType))
        end
        
        (precision[fold], tasa_error[fold], sensibilidad[fold], especificidad[fold], VPP[fold], VPN[fold], F1[fold], matriz_confusion[:, :, fold]) = confusionMatrix(testOutputs, testTargets, classes)

    end

    precision_result = (mean(precision), std(precision))
    tasa_error_result = (mean(tasa_error), std(tasa_error))
    sensibilidad_result = (mean(sensibilidad), std(sensibilidad))
    especificidad_result = (mean(especificidad), std(especificidad))
    VPP_result = (mean(VPP), std(VPP))
    VPN_result = (mean(VPN), std(VPN))
    F1_result = (mean(F1), std(F1))
    matriz_confusion_result = sum(matriz_confusion, dims=3)[:, :, 1]

    return precision_result, tasa_error_result, sensibilidad_result, especificidad_result, VPP_result, VPN_result, F1_result, matriz_confusion_result

end