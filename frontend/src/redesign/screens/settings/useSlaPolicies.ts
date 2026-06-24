/* ============================================================
   Data hook for Settings · SLA Policies — list + CRUD + set-default
   over slaPoliciesApi. Same useEffect + shared-axios + phase/reload
   pattern as the other settings hooks (see useSettings.ts). The
   action helpers reload the list on success. REDESIGN_GAPS.md §8.
   ============================================================ */
import { useCallback, useEffect, useState } from 'react'
import { slaPoliciesApi } from '../../../services/api'

export type Phase = 'loading' | 'ready' | 'error'
export type PriorityLevel = 'critical' | 'high' | 'medium' | 'low'

export interface SlaPolicy {
  policy_id: string
  name: string
  description?: string | null
  priority_level: PriorityLevel | string
  response_time_hours: number
  resolution_time_hours: number
  business_hours_only?: boolean
  notification_thresholds?: number[]
  is_active?: boolean
  is_default?: boolean
}

export interface SlaPolicyCreate {
  policy_id: string
  name: string
  description?: string
  priority_level: string
  response_time_hours: number
  resolution_time_hours: number
  business_hours_only?: boolean
  notification_thresholds?: number[]
  is_active?: boolean
  is_default?: boolean
}
export type SlaPolicyUpdate = Partial<Omit<SlaPolicyCreate, 'policy_id' | 'priority_level'>>

export function useSlaPolicies() {
  const [policies, setPolicies] = useState<SlaPolicy[]>([])
  const [phase, setPhase] = useState<Phase>('loading')
  const [error, setError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)
  const reload = useCallback(() => setReloadKey((k) => k + 1), [])

  useEffect(() => {
    let cancelled = false
    setPhase('loading')
    setError(null)
    slaPoliciesApi
      .getAll()
      .then((res) => {
        if (cancelled) return
        const data = res.data
        // tolerate both a bare array and a { policies: [...] } envelope
        const list = (Array.isArray(data) ? data : data?.policies || []) as SlaPolicy[]
        setPolicies(list)
        setPhase('ready')
      })
      .catch((e) => {
        if (cancelled) return
        setError((e as { message?: string })?.message || 'Failed to load SLA policies')
        setPhase('error')
      })
    return () => {
      cancelled = true
    }
  }, [reloadKey])

  const create = useCallback(
    async (data: SlaPolicyCreate) => { await slaPoliciesApi.create(data); reload() },
    [reload],
  )
  const update = useCallback(
    async (id: string, data: SlaPolicyUpdate) => { await slaPoliciesApi.update(id, data); reload() },
    [reload],
  )
  const remove = useCallback(
    async (id: string, force?: boolean) => { await slaPoliciesApi.delete(id, force); reload() },
    [reload],
  )
  const setDefault = useCallback(
    async (id: string) => {
      // Single-default-per-priority UI guard: picking a default instantly
      // clears every sibling at the same priority_level so the table can never
      // show two defaults for one priority (mirrors the backend invariant).
      // Reload on failure to restore the authoritative state.
      setPolicies((prev) => {
        const target = prev.find((p) => p.policy_id === id)
        if (!target) return prev
        return prev.map((p) =>
          p.priority_level === target.priority_level
            ? { ...p, is_default: p.policy_id === id }
            : p,
        )
      })
      try {
        await slaPoliciesApi.setDefault(id)
      } catch (e) {
        reload()
        throw e
      }
    },
    [reload],
  )

  return { policies, phase, error, reload, create, update, remove, setDefault }
}
