import Pkg
Pkg.add("CSV")
Pkg.update()
Pkg.add("DataFrames")
Pkg.add("DataStructures")
Pkg.add("TextAnalysis")
using DataFrames
using CSV
using TextAnalysis, DataStructures
using Dates
using HDF5, JLD
import Dates
using MPI
MPI.Init()

startTime = Dates.Time(Dates.now()) 

df = DataFrame(CSV.File("../Data/amazon_reviews_us_Books_v1_01.tsv"))

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
mpi_process=MPI.Comm_size(comm)

data_each_process_contains=nrow(df)/mpi_process

stInd = floor(Int, (data_each_process_contains*rank)+1)
endInd = floor(Int, data_each_process_contains*(rank+1))
println("Process ", rank, " startIndex ", stInd, " and endIndex ", endInd)

docs=[]
for i in stInd:endInd
    if typeof(df.review_body[i])==String
        sd=TokenDocument(df.review_body[i])
        prepare!(sd, strip_stopwords)
        prepare!(sd, strip_non_letters)
        push!(docs, sd) 
    end
end

println("Process ", rank," length of docs: ", length(docs))
termDict=Dict()
docTerm=Dict()
docTermCount=[]
for i in 1:length(docs)
    dt=Dict()
    for t in docs[i].tokens
        if haskey(termDict, t)
            termDict[t]=termDict[t]+1
        else
            termDict[t]=1
        end
        if haskey(docTerm, t)
            push!(docTerm[t], i)
        else
            docTerm[t]=Set(i)
        end
        if haskey(dt, t)
            dt[t]=dt[t]+1
        else
            dt[t]=1
        end
    end
    push!(docTermCount, dt)
end

termDocCount=Dict()
for (k,v) in docTerm
    termDocCount[k]=length(v)
end
termDocCount["docLength"]=length(docs)

save(string("process", rank, ".jld2"), "data", termDocCount)

MPI.Barrier(comm)

totalDocLength=0
for i in 0:mpi_process
    temp=load(string("process", rank, ".jld2"))["data"]
    global totalDocLength += temp["docLength"]
    delete!(temp,"docLength" )
    for (k,v) in temp
        if haskey(termDocCount, k)
            termDocCount[k]+=v
        else
            termDocCount[k]=v
        end
    end
end

IDF=Dict()
for (k,v) in termDocCount
    IDF[k]=log2(totalDocLength/v)
end

tf_idf=[]
for i in 1:length(docs)
    dt=Dict()
    for (k,v) in docTermCount[i]
        dt[k]=v*IDF[k]
    end
    push!(tf_idf, dt)
end

MPI.Barrier(comm)

endTime = Dates.Time(Dates.now()) 

println("Time Taken: ", endTime-startTime)