{{/*
Expand the name of the chart.
*/}}
{{- define "deep-researcher.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "deep-researcher.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "deep-researcher.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "deep-researcher.labels" -}}
helm.sh/chart: {{ include "deep-researcher.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
project: {{ .Values.project.name }}
{{- with .Values.project.deploymentEnv }}
environment: {{ . }}
{{- end }}
{{- if ne (.Values.project.orientation | default "internal") "external" }}
{{- with .Values.project.dl }}
dl: {{ . }}
{{- end }}
{{- with .Values.project.nspect_id }}
nspect_id: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Generate namespace name

Always uses the release namespace (`helm install -n <ns>` / `.Release.Namespace`)
so the chart honors the install namespace and works with GitOps operators such as
ArgoCD and Fleet that do not pre-create a `ns-<appname>` namespace.
*/}}
{{- define "deep-researcher.namespace" -}}
{{- .Release.Namespace -}}
{{- end }}

{{/*
Generate app-specific name
*/}}
{{- define "deep-researcher.appName" -}}
{{- $global := index . 0 -}}
{{- $appName := index . 1 -}}
{{- printf "%s-%s" $global.Values.project.name $appName | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Generate app-specific fullname (with truncation)
*/}}
{{- define "deep-researcher.appFullname" -}}
{{- $global := index . 0 -}}
{{- $appName := index . 1 -}}
{{- printf "%s-%s" $global.Values.project.name ($appName | trunc 15 | trimSuffix "-") | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Generate app-specific labels
*/}}
{{- define "deep-researcher.appLabels" -}}
{{- $global := index . 0 -}}
{{- $appName := index . 1 -}}
{{- $appConfig := index . 2 -}}
{{ include "deep-researcher.labels" $global }}
app: {{ include "deep-researcher.appName" (list $global $appName) }}
{{- with $appConfig.labels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Generate app-specific selector labels
*/}}
{{- define "deep-researcher.appSelectorLabels" -}}
{{- $global := index . 0 -}}
{{- $appName := index . 1 -}}
app: {{ include "deep-researcher.appName" (list $global $appName) }}
{{- end }}

{{/*
Get component name (auto-derive from app name if not specified)
*/}}
{{- define "deep-researcher.component" -}}
{{- $appName := index . 0 -}}
{{- $appConfig := index . 1 -}}
{{- $appConfig.component | default $appName -}}
{{- end }}

{{/*
Get image repository (use global, appDefaults, or app-specific)
*/}}
{{- define "deep-researcher.imageRepository" -}}
{{- $global := index . 0 -}}
{{- $appConfig := index . 1 -}}
{{- $defaults := $global.Values.appDefaults | default dict -}}
{{- $defaultImage := $defaults.image | default dict -}}
{{- $appImage := $appConfig.image | default dict -}}
{{- if $appImage.repository -}}
{{- $appImage.repository -}}
{{- else if $global.Values.imageRepository -}}
{{- $global.Values.imageRepository -}}
{{- else if $defaultImage.repository -}}
{{- $defaultImage.repository -}}
{{- else -}}
{{- fail "image.repository is required. Set it globally via imageRepository, in appDefaults.image.repository, or per app" -}}
{{- end -}}
{{- end }}

{{/*
Extract base name for auto-generating image names
Uses project.name as the base for image names (e.g., "test-francis-z-2")
This ensures image names match the project naming convention
*/}}
{{- define "deep-researcher.imageBaseName" -}}
{{- $global := index . 0 -}}
{{- $appConfig := index . 1 -}}
{{- $global.Values.project.name -}}
{{- end }}

{{/*
Get image name (auto-generate from imageRepository base + app if not specified)
*/}}
{{- define "deep-researcher.imageName" -}}
{{- $global := index . 0 -}}
{{- $appName := index . 1 -}}
{{- $appConfig := index . 2 -}}
{{- $appImage := $appConfig.image | default dict -}}
{{- if $appImage.name -}}
{{- $appImage.name -}}
{{- else -}}
{{- $baseName := include "deep-researcher.imageBaseName" (list $global $appConfig) -}}
{{- printf "%s-%s" $baseName $appName -}}
{{- end -}}
{{- end }}

{{/*
Auto-generate DataDog labels from project metadata (no-op if customerID not set)
*/}}
{{- define "deep-researcher.datadogLabels" -}}
{{- end }}

{{/*
Get app port (from port or ports[0].containerPort)
*/}}
{{- define "deep-researcher.appPort" -}}
{{- $appConfig := index . 0 -}}
{{- if hasKey $appConfig "port" -}}
{{- $appConfig.port -}}
{{- else if hasKey $appConfig "ports" -}}
{{- if $appConfig.ports -}}
{{- (index $appConfig.ports 0).containerPort -}}
{{- else -}}
{{- fail "Either 'port' or 'ports' must be specified for the app" -}}
{{- end -}}
{{- else -}}
{{- fail "Either 'port' or 'ports' must be specified for the app" -}}
{{- end -}}
{{- end }}

{{/*
Auto-generate ingress hostname
Uses base name from imageRepository instead of full project name for consistency
*/}}
{{- define "deep-researcher.ingressHostname" -}}
{{- $global := index . 0 -}}
{{- $appName := index . 1 -}}
{{- $appConfig := index . 2 -}}
{{- $appIngress := $appConfig.ingress | default dict -}}
{{- $baseName := include "deep-researcher.imageBaseName" (list $global $appConfig) -}}
{{- if eq (kindOf $appIngress) "map" -}}
{{- if $appIngress.hostname -}}
{{- $appIngress.hostname -}}
{{- else -}}
{{- printf "%s-%s.local" $baseName $appName -}}
{{- end -}}
{{- else -}}
{{- printf "%s-%s.local" $baseName $appName -}}
{{- end -}}
{{- end }}

{{/*
Merge app config with defaults
Usage: include "deep-researcher.mergeAppDefaults" (list $ $appName $appConfig)
Returns: merged config with defaults applied
*/}}
{{- define "deep-researcher.mergeAppDefaults" -}}
{{- $global := index . 0 -}}
{{- $appName := index . 1 -}}
{{- $appConfig := index . 2 -}}
{{- $merged := deepCopy $global.Values.appDefaults | mustMerge (deepCopy $appConfig) -}}
{{- $merged | toJson -}}
{{- end }}

{{/*
Get app config value with default fallback
Usage: include "deep-researcher.appValue" (dict "global" $ "app" $appConfig "path" "image.pullPolicy" "default" "IfNotPresent")
*/}}
{{- define "deep-researcher.appValue" -}}
{{- $global := .global -}}
{{- $app := .app -}}
{{- $path := .path -}}
{{- $default := .default -}}
{{- $pathParts := splitList "." $path -}}
{{- $appValue := $app -}}
{{- $defaultValue := $global.Values.appDefaults -}}
{{- $foundApp := true -}}
{{- $foundDefault := true -}}
{{- range $pathParts -}}
  {{- if and $foundApp (hasKey $appValue .) -}}
    {{- $appValue = index $appValue . -}}
  {{- else -}}
    {{- $foundApp = false -}}
  {{- end -}}
  {{- if and $foundDefault (hasKey $defaultValue .) -}}
    {{- $defaultValue = index $defaultValue . -}}
  {{- else -}}
    {{- $foundDefault = false -}}
  {{- end -}}
{{- end -}}
{{- if $foundApp -}}
  {{- $appValue -}}
{{- else if $foundDefault -}}
  {{- $defaultValue -}}
{{- else -}}
  {{- $default -}}
{{- end -}}
{{- end }}
